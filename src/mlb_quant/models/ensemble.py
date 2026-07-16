"""Ensemble calibrado de moneyline.

Tres miembros complementarios — probabilidad del Poisson (vía malla de
mercados), regresión logística y gradient boosting — combinados por una
"cabeza" ajustada sobre una cola temporal del entrenamiento (los
miembros no ven esos juegos: la combinación/calibración es honesta).

Dos cabezas disponibles:

- **Promedio + calibrador** (``stacking=False``): media simple de los
  miembros, calibrada con isotónica/Platt/beta (:mod:`.calibration`).
- **Stacking** (``stacking=True``): meta-logística sobre los logits de
  los miembros — aprende los pesos en vez de promediar. Su sigmoide ya
  es una calibración Platt multivariable, así que se combina con
  ``calibration="none"``; forzar un calibrador extra divide la cola en
  dos mitades (meta y calibrador) para no reusar los mismos juegos.

La separación miembros/cabeza permite comparar cabezas en el backtest
sin reentrenar miembros (:meth:`CalibratedEnsembleWinModel.with_head`).
"""

import logging

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mlb_quant.models.base import feature_columns, to_design_matrix
from mlb_quant.models.calibration import (
    CalibrationKind,
    Calibrator,
    IdentityCalibrator,
    _logit,
    make_calibrator,
)
from mlb_quant.models.logistic import LogisticWinModel
from mlb_quant.models.markets import market_probabilities
from mlb_quant.models.poisson import PoissonRunsModel

logger = logging.getLogger(__name__)

#: Hiperparámetros por defecto del boosting. early_stopping usa un split
#: interno aleatorio que vive completo dentro del núcleo de entrenamiento
#: (nunca toca la cola de calibración ni el mes evaluado del backtest).
HGB_DEFAULTS: dict[str, float | int | bool] = {
    "max_iter": 1000,
    "learning_rate": 0.05,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}


class CalibratedEnsembleWinModel:
    """Poisson + logística + boosting con cabeza intercambiable.

    Attributes:
        calibration_fraction: Fracción final (temporal) del entrenamiento
            reservada para ajustar la cabeza.
        stacking: Si la cabeza es meta-logística (True) o promedio (False).
        calibration: Calibrador aplicado a la salida de la cabeza.
    """

    def __init__(
        self,
        calibration_fraction: float = 0.2,
        seed: int = 42,
        *,
        stacking: bool = False,
        calibration: CalibrationKind = "isotonic",
        hgb_params: dict[str, float | int | bool] | None = None,
    ) -> None:
        """Inicializa el ensemble sin entrenar.

        Args:
            calibration_fraction: Cola temporal para la cabeza (0.05-0.5).
            seed: Semilla del boosting.
            stacking: Meta-logística en vez de promedio simple.
            calibration: ``isotonic`` | ``platt`` | ``beta`` | ``none``.
            hgb_params: Sobrescrituras de :data:`HGB_DEFAULTS`.
        """
        if not 0.05 <= calibration_fraction <= 0.5:
            msg = "calibration_fraction debe estar entre 0.05 y 0.5."
            raise ValueError(msg)
        make_calibrator(calibration)  # valida el nombre temprano
        self.calibration_fraction = calibration_fraction
        self.stacking = stacking
        self.calibration: CalibrationKind = calibration
        self._seed = seed
        self._hgb_params = HGB_DEFAULTS | (hgb_params or {})
        self._columns: list[str] = []
        self._poisson = PoissonRunsModel()
        self._logistic = LogisticWinModel()
        self._boosting = HistGradientBoostingClassifier(
            random_state=seed, **self._hgb_params
        )
        self._meta: LogisticRegression | None = None
        self._calibrator: Calibrator = IdentityCalibrator()
        self._tail_member_probs: np.ndarray | None = None
        self._tail_y: np.ndarray | None = None

    def fit(self, features: pl.DataFrame) -> "CalibratedEnsembleWinModel":
        """Entrena miembros y cabeza con separación temporal.

        Los miembros se entrenan con el tramo inicial; la cabeza con la
        cola (juegos que los miembros nunca vieron).

        Args:
            features: Matriz con features y labels, cualquier orden.

        Returns:
            El propio modelo.

        Raises:
            ValueError: Si faltan labels o hay muy pocos juegos.
        """
        if "home_score" not in features.columns or "away_score" not in features.columns:
            msg = "features debe incluir home_score y away_score para entrenar."
            raise ValueError(msg)
        features = features.sort("game_date")
        n_calibration = max(int(len(features) * self.calibration_fraction), 50)
        if len(features) < 2 * n_calibration:
            msg = f"Muy pocos juegos ({len(features)}) para entrenar y calibrar."
            raise ValueError(msg)

        core = features.head(len(features) - n_calibration)
        tail = features.tail(n_calibration)

        self._fit_members(core)
        self._tail_member_probs = self._member_probabilities(tail)
        self._tail_y = (
            (tail["home_score"] > tail["away_score"]).to_numpy().astype(int)
        )
        self._fit_head()
        logger.info(
            "Ensemble entrenado: %d núcleo + %d cola (stacking=%s, calibración=%s).",
            len(core),
            n_calibration,
            self.stacking,
            self.calibration,
        )
        return self

    def with_head(
        self, *, stacking: bool, calibration: CalibrationKind
    ) -> "CalibratedEnsembleWinModel":
        """Copia ligera con otra cabeza sobre los mismos miembros entrenados.

        Comparte los miembros y la cola cacheada; solo reajusta la cabeza
        (milisegundos). Permite comparar configuraciones en el backtest
        sin repetir el entrenamiento caro.

        Args:
            stacking: Cabeza de la copia.
            calibration: Calibrador de la copia.

        Returns:
            Ensemble nuevo listo para predecir; el original no se muta.

        Raises:
            ValueError: Si este ensemble aún no fue entrenado.
        """
        if self._tail_member_probs is None or self._tail_y is None:
            msg = "with_head requiere un ensemble entrenado (llama fit primero)."
            raise ValueError(msg)
        clone = CalibratedEnsembleWinModel(
            self.calibration_fraction,
            seed=self._seed,
            stacking=stacking,
            calibration=calibration,
            hgb_params=dict(self._hgb_params),
        )
        clone._columns = self._columns
        clone._poisson = self._poisson
        clone._logistic = self._logistic
        clone._boosting = self._boosting
        clone._tail_member_probs = self._tail_member_probs
        clone._tail_y = self._tail_y
        clone._fit_head()
        return clone

    def predict_home_win(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predice P(gana el local) calibrada.

        Args:
            features: Matriz de features (con ``game_pk``).

        Returns:
            ``game_pk``, ``p_home_win``.
        """
        raw = self._head_probabilities(self._member_probabilities(features))
        calibrated = self._calibrator.predict(raw)
        return pl.DataFrame(
            {"game_pk": features["game_pk"], "p_home_win": calibrated}
        )

    def _fit_members(self, core: pl.DataFrame) -> None:
        """Entrena los tres miembros con el núcleo temporal."""
        # Columnas sin variación (todo null o constantes) rompen el
        # binning del boosting y no aportan señal: fuera.
        self._columns = [
            c
            for c in feature_columns(core)
            if core[c].drop_nulls().n_unique() >= 2
        ]
        self._poisson.fit(core)
        self._logistic.fit(core)
        x_core = to_design_matrix(core, self._columns)
        y_core = (core["home_score"] > core["away_score"]).to_numpy()
        self._boosting.fit(x_core, y_core)

    def _fit_head(self) -> None:
        """Ajusta la cabeza (meta y/o calibrador) sobre la cola cacheada."""
        assert self._tail_member_probs is not None and self._tail_y is not None
        probs, y = self._tail_member_probs, self._tail_y

        if not self.stacking:
            self._meta = None
            raw = probs.mean(axis=1)
            self._calibrator = make_calibrator(self.calibration).fit(raw, y)
            return

        if self.calibration == "none":
            meta_probs, meta_y = probs, y
            calib_probs = calib_y = None
        else:
            # Doble ajuste sobre la misma cola sesgaría el calibrador:
            # mitad temporal para la meta, mitad para el calibrador.
            logger.warning(
                "stacking + calibración %s: cola dividida 50/50 (meta / calibrador). "
                "Con colas chicas prefiere calibration='none'.",
                self.calibration,
            )
            half = len(probs) // 2
            meta_probs, meta_y = probs[:half], y[:half]
            calib_probs, calib_y = probs[half:], y[half:]

        self._meta = LogisticRegression(max_iter=2000)
        self._meta.fit(_logit(meta_probs), meta_y)
        if calib_probs is None or calib_y is None:
            self._calibrator = IdentityCalibrator()
        else:
            raw = self._meta.predict_proba(_logit(calib_probs))[:, 1]
            self._calibrator = make_calibrator(self.calibration).fit(raw, calib_y)

    def _head_probabilities(self, member_probs: np.ndarray) -> np.ndarray:
        """Salida de la cabeza (sin calibrador) para probs de miembros (n, 3)."""
        if self._meta is not None:
            return np.asarray(
                self._meta.predict_proba(_logit(member_probs))[:, 1], dtype=np.float64
            )
        return np.asarray(member_probs.mean(axis=1), dtype=np.float64)

    def _member_probabilities(self, features: pl.DataFrame) -> np.ndarray:
        """Probabilidades de los tres miembros, matriz (n, 3)."""
        p_poisson = market_probabilities(self._poisson.predict_lambdas(features))[
            "p_home_ml"
        ].to_numpy()
        p_logistic = self._logistic.predict_home_win(features)["p_home_win"].to_numpy()
        x = to_design_matrix(features, self._columns)
        p_boosting = self._boosting.predict_proba(x)[:, 1]
        return np.column_stack(
            [
                np.asarray(p_poisson, dtype=np.float64),
                np.asarray(p_logistic, dtype=np.float64),
                np.asarray(p_boosting, dtype=np.float64),
            ]
        )
