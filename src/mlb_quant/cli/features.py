"""Subcomandos ``mlb features``: construcción de matrices de features."""

import logging
from datetime import date, datetime

import typer
from rich.console import Console

import mlb_quant.feature_engineering.blocks  # noqa: F401  (registra los bloques)
from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import get_blocks
from mlb_quant.feature_engineering.builder import build_game_features
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

features_app = typer.Typer(help="Construye matrices de features desde el warehouse.")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"Fecha inválida: {value!r}. Formato: YYYY-MM-DD.") from exc


@features_app.command()
def build(
    start: str = typer.Option(..., help="Fecha inicial (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Fecha final (YYYY-MM-DD). Por defecto, igual a start."),
    blocks: str = typer.Option(None, help="Bloques separados por coma. Por defecto, todos."),
) -> None:
    """Construye features por juego -> tabla ``features_game``."""
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date
    if end_date < start_date:
        raise typer.BadParameter("end no puede ser anterior a start.")
    block_names = [b.strip() for b in blocks.split(",")] if blocks else None

    db = Database(get_settings().duckdb_path)
    try:
        df = build_game_features(db, start_date, end_date, block_names)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if df.is_empty():
        console.print("[yellow]Sin juegos finalizados en el rango.[/yellow]")
        return
    written = db.upsert("features_game", df, keys=["game_pk"])
    console.print(
        f"[green]features_game:[/green] {written} juegos, {len(df.columns)} columnas "
        f"({start_date} -> {end_date})."
    )


@features_app.command(name="list")
def list_blocks() -> None:
    """Lista los bloques de features registrados."""
    for block in get_blocks():
        console.print(
            f"[cyan]{block.name}[/cyan] (grano {block.grain}) "
            f"- requiere: {', '.join(block.requires)}"
        )
