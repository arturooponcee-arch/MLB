"""Modelo Poisson de carreras: base interpretable del sistema.

Dos regresiones Poisson independientes (carreras del local y del
visitante) sobre las features del juego. De sus tasas (lambdas) se
derivan todos los mercados de equipo vía :mod:`mlb_quant.models.markets`.

La independencia entre lambdas es una aproximación conocida (subestima
ligeramente la correlación por clima/parque compartido, ya parcialmente
capturada porque ambas regresiones ven las mismas features de contexto).
"""

import logging

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_quant.models.base import feature_columns, to_design_matrix

logger = logging.getLogger(__name__)


def _make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000)),
        ]
    )


class PoissonRunsModel:
    """Regresión Poisson de carreras por lado (home/away).

    Attributes:
        alpha: Regularización L2 de ambas regresiones.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Inicializa el modelo sin entrenar.

        Args:
            alpha: Regularización L2 (mayor = más conservador).
        """
        self.alpha = alpha
        self._columns: list[str] = []
        self._home = _make_pipeline(alpha)
        self._away = _make_pipeline(alpha)

    def fit(self, features: pl.DataFrame) -> "PoissonRunsModel":
        """Entrena ambas regresiones sobre juegos con marcador conocido.

        Args:
            features: Matriz con features y labels (``home_score``,
                ``away_score``).

        Returns:
            El propio modelo (para encadenar).

        Raises:
            ValueError: Si la matriz no trae los labels.
        """
        if "home_score" not in features.columns or "away_score" not in features.columns:
            msg = "features debe incluir home_score y away_score para entrenar."
            raise ValueError(msg)
        self._columns = feature_columns(features)
        x = to_design_matrix(features, self._columns)
        self._home.fit(x, features["home_score"].to_numpy())
        self._away.fit(x, features["away_score"].to_numpy())
        logger.info(
            "PoissonRunsModel entrenado: %d juegos, %d features.",
            len(features),
            len(self._columns),
        )
        return self

    def coefficients(self, side: str = "home") -> pl.DataFrame:
        """Coeficientes estandarizados de una de las regresiones.

        Las features se escalan antes de la regresión, así que los
        coeficientes son comparables entre sí (efecto en log-carreras por
        desviación estándar).

        Args:
            side: ``home`` o ``away``.

        Returns:
            ``feature``, ``coefficient``, ordenado por efecto descendente.

        Raises:
            ValueError: Si ``side`` es inválido o el modelo no está entrenado.
        """
        if side not in ("home", "away"):
            msg = f"side debe ser home o away, no {side!r}."
            raise ValueError(msg)
        if not self._columns:
            msg = "Modelo sin entrenar."
            raise ValueError(msg)
        pipeline = self._home if side == "home" else self._away
        return pl.DataFrame(
            {
                "feature": self._columns,
                "coefficient": pipeline.named_steps["poisson"].coef_,
            }
        ).sort("coefficient", descending=True)

    def predict_lambdas(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predice la tasa esperada de carreras de cada lado.

        Args:
            features: Matriz de features (con ``game_pk``).

        Returns:
            ``game_pk``, ``lambda_home``, ``lambda_away``.
        """
        x = to_design_matrix(features, self._columns)
        lambda_home = np.asarray(self._home.predict(x), dtype=np.float64)
        lambda_away = np.asarray(self._away.predict(x), dtype=np.float64)
        return pl.DataFrame(
            {
                "game_pk": features["game_pk"],
                "lambda_home": np.clip(lambda_home, 0.05, None),
                "lambda_away": np.clip(lambda_away, 0.05, None),
            }
        )
