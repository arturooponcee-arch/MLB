"""Ensemble calibrado de moneyline.

Promedia tres miembros complementarios — probabilidad del Poisson (vía
malla de mercados), regresión logística y gradient boosting — y calibra
el promedio con regresión isotónica ajustada sobre una cola temporal del
entrenamiento (los miembros no ven esos juegos: la calibración es
honesta).
"""

import logging

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from mlb_quant.models.base import feature_columns, to_design_matrix
from mlb_quant.models.logistic import LogisticWinModel
from mlb_quant.models.markets import market_probabilities
from mlb_quant.models.poisson import PoissonRunsModel

logger = logging.getLogger(__name__)


class CalibratedEnsembleWinModel:
    """Promedio de Poisson + logística + boosting, calibrado isotónico.

    Attributes:
        calibration_fraction: Fracción final (temporal) del entrenamiento
            reservada para ajustar el calibrador.
    """

    def __init__(self, calibration_fraction: float = 0.2, seed: int = 42) -> None:
        """Inicializa el ensemble sin entrenar.

        Args:
            calibration_fraction: Cola temporal para calibrar (0-0.5).
            seed: Semilla del boosting.
        """
        if not 0.05 <= calibration_fraction <= 0.5:
            msg = "calibration_fraction debe estar entre 0.05 y 0.5."
            raise ValueError(msg)
        self.calibration_fraction = calibration_fraction
        self._columns: list[str] = []
        self._poisson = PoissonRunsModel()
        self._logistic = LogisticWinModel()
        self._boosting = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed
        )
        self._calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )

    def fit(self, features: pl.DataFrame) -> "CalibratedEnsembleWinModel":
        """Entrena miembros y calibrador con separación temporal.

        Los miembros se entrenan con el tramo inicial; el calibrador con
        la cola (juegos que los miembros nunca vieron).

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
        calibration = features.tail(n_calibration)

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

        raw = self._raw_probabilities(calibration)
        y_calibration = (
            (calibration["home_score"] > calibration["away_score"])
            .to_numpy()
            .astype(int)
        )
        self._calibrator.fit(raw, y_calibration)
        logger.info(
            "Ensemble entrenado: %d núcleo + %d calibración.",
            len(core),
            n_calibration,
        )
        return self

    def predict_home_win(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predice P(gana el local) calibrada.

        Args:
            features: Matriz de features (con ``game_pk``).

        Returns:
            ``game_pk``, ``p_home_win``.
        """
        calibrated = self._calibrator.predict(self._raw_probabilities(features))
        return pl.DataFrame(
            {"game_pk": features["game_pk"], "p_home_win": calibrated}
        )

    def _raw_probabilities(self, features: pl.DataFrame) -> np.ndarray:
        """Promedio simple de los tres miembros (sin calibrar)."""
        p_poisson = market_probabilities(self._poisson.predict_lambdas(features))[
            "p_home_ml"
        ].to_numpy()
        p_logistic = self._logistic.predict_home_win(features)["p_home_win"].to_numpy()
        x = to_design_matrix(features, self._columns)
        p_boosting = self._boosting.predict_proba(x)[:, 1]
        return np.asarray((p_poisson + p_logistic + p_boosting) / 3.0, dtype=np.float64)
