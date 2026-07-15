"""Exportación de DataFrames a CSV, Excel y Markdown.

PDF pendiente: exigiría weasyprint/reportlab (dependencias pesadas); el
Markdown se convierte con pandoc si hace falta.
"""

import logging
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

type ExportFormat = Literal["csv", "xlsx", "md"]

DEFAULT_FORMATS: tuple[ExportFormat, ...] = ("csv", "xlsx", "md")


def export_frame(
    df: pl.DataFrame,
    base_path: Path,
    formats: tuple[ExportFormat, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Escribe ``df`` en cada formato junto a ``base_path``.

    Args:
        df: Datos a exportar.
        base_path: Ruta sin extensión (``reports/daily/2026-07-17``).
        formats: Formatos a generar.

    Returns:
        Rutas escritas.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        if fmt == "csv":
            df.write_csv(path)
        elif fmt == "xlsx":
            df.write_excel(path)
        else:
            path.write_text(to_markdown(df), encoding="utf-8")
        written.append(path)
    logger.info("Exportado %s (%d filas) a %s.", base_path.name, len(df), formats)
    return written


def to_markdown(df: pl.DataFrame, max_rows: int = 500) -> str:
    """Convierte un DataFrame en tabla Markdown.

    Args:
        df: Datos a renderizar.
        max_rows: Tope de filas (evita reportes gigantes).

    Returns:
        Tabla en sintaxis GitHub-flavored Markdown.
    """
    if df.is_empty():
        return "_Sin datos._\n"
    df = df.head(max_rows)
    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "|".join(" --- " for _ in df.columns) + "|"
    lines = [header, divider]
    for row in df.iter_rows():
        cells = ["" if v is None else _format_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")
