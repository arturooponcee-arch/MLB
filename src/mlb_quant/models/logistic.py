"""Baseline logístico de moneyline.

Modelo directo de P(gana el local). Sirve de sanity check del Poisson:
si la logística supera claramente al Poisson en log loss, el Poisson
está mal especificado.
"""

import logging

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_quant.models.base import feature_columns, to_design_matrix

logger = logging.getLogger(__name__)


class LogisticWinModel:
    """Regresión logística de victoria del equipo local."""

    def __init__(self, c: float = 1.0) -> None:
        """Inicializa el modelo sin entrenar.

        Args:
            c: Inverso de la regularización (menor = más conservador).
        """
        self._columns: list[str] = []
        self._pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(C=c, max_iter=2000)),
            ]
        )

    def fit(self, features: pl.DataFrame) -> "LogisticWinModel":
        """Entrena sobre juegos con marcador conocido.

        Args:
            features: Matriz con features y labels.

        Returns:
            El propio modelo.

        Raises:
            ValueError: Si faltan los labels.
        """
        if "home_score" not in features.columns or "away_score" not in features.columns:
            msg = "features debe incluir home_score y away_score para entrenar."
            raise ValueError(msg)
        self._columns = feature_columns(features)
        x = to_design_matrix(features, self._columns)
        y = (features["home_score"] > features["away_score"]).to_numpy()
        self._pipeline.fit(x, y)
        logger.info(
            "LogisticWinModel entrenado: %d juegos, %d features.",
            len(features),
            len(self._columns),
        )
        return self

    def predict_home_win(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predice P(gana el local) por juego.

        Args:
            features: Matriz de features (con ``game_pk``).

        Returns:
            ``game_pk``, ``p_home_win``.
        """
        x = to_design_matrix(features, self._columns)
        proba = np.asarray(self._pipeline.predict_proba(x)[:, 1], dtype=np.float64)
        return pl.DataFrame({"game_pk": features["game_pk"], "p_home_win": proba})
