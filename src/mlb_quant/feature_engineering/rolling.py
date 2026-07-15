"""Utilidades de rolling/lag sin fuga de información.

Toda agregación aplica ``shift(1)`` dentro del grupo ANTES de la ventana:
la fila de un juego solo ve juegos estrictamente anteriores. Nunca usar
rolling sin lag sobre datos que incluyan el juego a predecir.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class RollingSpec:
    """Especificación de una feature rolling.

    Attributes:
        column: Columna de origen.
        window: Tamaño de la ventana (en filas, no días).
        agg: Agregación a aplicar.
        alias: Nombre de la columna resultante.
    """

    column: str
    window: int
    agg: Literal["mean", "sum"]
    alias: str


def add_lagged_rolling(
    df: pl.DataFrame,
    *,
    group_by: str | Sequence[str],
    sort_by: str | Sequence[str],
    specs: Sequence[RollingSpec],
) -> pl.DataFrame:
    """Añade columnas rolling con lag de 1 fila dentro de cada grupo.

    La primera fila de cada grupo queda en null (sin historia previa).

    Args:
        df: Datos en formato largo (una fila por evento).
        group_by: Columna(s) que identifican la entidad (equipo, pitcher).
        sort_by: Columna(s) de orden temporal dentro del grupo.
        specs: Features rolling a generar.

    Returns:
        ``df`` ordenado por grupo y tiempo, con una columna por spec.
    """
    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)

    expressions = []
    for spec in specs:
        lagged = pl.col(spec.column).shift(1)
        if spec.agg == "mean":
            rolled = lagged.rolling_mean(window_size=spec.window, min_samples=1)
        else:
            rolled = lagged.rolling_sum(window_size=spec.window, min_samples=1)
        expressions.append(rolled.over(group_cols).alias(spec.alias))

    return df.sort(group_cols + sort_cols).with_columns(expressions)


def add_days_since_previous(
    df: pl.DataFrame,
    *,
    group_by: str | Sequence[str],
    date_column: str,
    alias: str,
) -> pl.DataFrame:
    """Añade los días transcurridos desde la fila anterior del grupo.

    Null en la primera aparición de cada entidad.

    Args:
        df: Datos en formato largo.
        group_by: Columna(s) que identifican la entidad.
        date_column: Columna de fecha (``pl.Date``).
        alias: Nombre de la columna resultante.

    Returns:
        ``df`` ordenado, con la columna de descanso añadida.
    """
    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    return df.sort([*group_cols, date_column]).with_columns(
        (pl.col(date_column) - pl.col(date_column).shift(1))
        .dt.total_days()
        .over(group_cols)
        .alias(alias)
    )
