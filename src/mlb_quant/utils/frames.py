"""Utilidades de DataFrames compartidas entre fuentes."""

import re

import pandas as pd

_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("%", "_pct"),
    ("+", "_plus"),
    ("/", "_"),
    ("-", "_"),
    (" ", "_"),
    (".", "_"),
    ("(", ""),
    (")", ""),
)


def sanitize_column(name: str) -> str:
    """Convierte un nombre de columna a snake_case seguro para SQL.

    Ejemplos: ``K%`` -> ``k_pct``, ``wRC+`` -> ``wrc_plus``,
    ``HR/FB`` -> ``hr_fb``, ``1B`` -> ``x1b``.

    Args:
        name: Nombre original.

    Returns:
        Nombre saneado.
    """
    result = name.strip()
    for old, new in _REPLACEMENTS:
        result = result.replace(old, new)
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = f"x{result}"
    return result


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de ``df`` con todas las columnas saneadas.

    Args:
        df: DataFrame con nombres de columna arbitrarios (ej. FanGraphs).

    Returns:
        DataFrame con columnas snake_case únicas.

    Raises:
        ValueError: Si el saneado produce nombres duplicados.
    """
    mapping = {col: sanitize_column(str(col)) for col in df.columns}
    sanitized = list(mapping.values())
    duplicates = {name for name in sanitized if sanitized.count(name) > 1}
    if duplicates:
        msg = f"Columnas duplicadas tras sanear: {sorted(duplicates)}"
        raise ValueError(msg)
    return df.rename(columns=mapping)
