"""Utilidades comunes de modelado.

Convención: la matriz ``features_game`` trae columnas meta (claves y
labels) que NUNCA entran al modelo; el resto de columnas numéricas son
features. Los modelos guardan la lista de features con la que se
entrenaron y la reutilizan al predecir (robustez ante columnas nuevas).
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import polars as pl

#: Columnas que nunca son features (claves y labels).
META_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "season",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Devuelve las columnas de features (numéricas, no meta) de una matriz."""
    return [
        c
        for c in df.columns
        if c not in META_COLUMNS and df.schema[c].is_numeric()
    ]


def to_design_matrix(df: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Extrae la matriz de diseño para sklearn.

    Columnas ausentes en ``df`` se rellenan con null (el imputador del
    pipeline las maneja); el orden queda fijado por ``columns``.

    Args:
        df: Matriz de features (Polars).
        columns: Features en el orden con que se entrenó el modelo.

    Returns:
        DataFrame pandas listo para sklearn.
    """
    present = [c for c in columns if c in df.columns]
    x = df.select(present).to_pandas() if present else pd.DataFrame(index=range(df.height))
    for missing in (c for c in columns if c not in x.columns):
        x[missing] = float("nan")
    return x[columns]


def save_model(model: Any, path: Path) -> None:
    """Serializa un modelo con joblib, creando el directorio si hace falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Carga un modelo serializado con joblib."""
    return joblib.load(path)
