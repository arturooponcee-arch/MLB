"""Subcomandos ``mlb ingest``: descarga de datos a DuckDB."""

import logging
from datetime import date, datetime

import typer
from rich.console import Console

from mlb_quant.db import Database
from mlb_quant.ingestion.mlb_stats_api import MlbStatsApi
from mlb_quant.ingestion.statcast import PITCH_KEYS, StatcastSource
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

ingest_app = typer.Typer(help="Descarga datos de fuentes externas a DuckDB.")


def _parse_date(value: str) -> date:
    """Convierte ``YYYY-MM-DD`` en ``date`` con error legible."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"Fecha inválida: {value!r}. Formato: YYYY-MM-DD.") from exc


def _database() -> Database:
    return Database(get_settings().duckdb_path)


@ingest_app.command()
def schedule(
    start: str = typer.Option(..., help="Fecha inicial (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Fecha final (YYYY-MM-DD). Por defecto, igual a start."),
) -> None:
    """Descarga calendario, probables y marcadores -> tabla ``games``."""
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date
    if end_date < start_date:
        raise typer.BadParameter("end no puede ser anterior a start.")

    df = MlbStatsApi().fetch_schedule(start_date, end_date)
    written = _database().upsert("games", df, keys=["game_pk"])
    console.print(f"[green]games:[/green] {written} filas escritas ({start_date} -> {end_date}).")


@ingest_app.command()
def teams(
    season: int = typer.Option(..., help="Temporada (año)."),
) -> None:
    """Descarga los equipos de una temporada -> tabla ``teams``."""
    df = MlbStatsApi().fetch_teams(season)
    written = _database().upsert("teams", df, keys=["team_id", "season"])
    console.print(f"[green]teams:[/green] {written} filas escritas (temporada {season}).")


@ingest_app.command()
def statcast(
    start: str = typer.Option(..., help="Fecha inicial (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Fecha final (YYYY-MM-DD). Por defecto, igual a start."),
) -> None:
    """Descarga pitch-by-pitch de Statcast -> tabla ``statcast_pitches``.

    Requiere: poetry install --with data
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date
    if end_date < start_date:
        raise typer.BadParameter("end no puede ser anterior a start.")

    try:
        df = StatcastSource().fetch_pitches(start_date, end_date)
    except ImportError as exc:
        console.print(
            "[red]pybaseball no instalado.[/red] Ejecuta: poetry install --with data"
        )
        raise typer.Exit(code=1) from exc

    if df.empty:
        console.print("[yellow]Sin datos en el rango (¿día sin juegos?).[/yellow]")
        return
    written = _database().upsert("statcast_pitches", df, keys=list(PITCH_KEYS))
    console.print(
        f"[green]statcast_pitches:[/green] {written} filas escritas "
        f"({start_date} -> {end_date})."
    )
