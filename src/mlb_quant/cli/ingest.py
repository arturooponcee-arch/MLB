"""Subcomandos ``mlb ingest``: descarga de datos a DuckDB."""

import logging
from datetime import date, datetime

import polars as pl
import typer
from rich.console import Console

from mlb_quant.db import Database
from mlb_quant.ingestion.lahman import LAHMAN_TABLES, LahmanSource
from mlb_quant.ingestion.mlb_stats_api import MlbStatsApi
from mlb_quant.ingestion.odds_api import ODDS_KEYS, OddsApiSource
from mlb_quant.ingestion.savant import SAVANT_KEYS, PlayerType, SavantLeaderboards
from mlb_quant.ingestion.statcast import PITCH_KEYS, StatcastSource
from mlb_quant.ingestion.weather import WeatherSource
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


@ingest_app.command()
def venues(
    season: int = typer.Option(..., help="Temporada (año)."),
) -> None:
    """Descarga estadios (coordenadas, dimensiones, techo) -> tabla ``venues``."""
    df = MlbStatsApi().fetch_venues(season)
    written = _database().upsert("venues", df, keys=["venue_id", "season"])
    console.print(f"[green]venues:[/green] {written} filas escritas (temporada {season}).")


@ingest_app.command()
def savant(
    season: int = typer.Option(..., help="Temporada (año)."),
    player_type: str = typer.Option("both", help="batter | pitcher | both."),
    min_pa: int = typer.Option(25, help="Mínimo de PA / bateadores enfrentados."),
) -> None:
    """Descarga leaderboards de Baseball Savant -> tablas ``savant_*``."""
    if player_type not in ("batter", "pitcher", "both"):
        raise typer.BadParameter("player_type debe ser batter, pitcher o both.")
    types: list[PlayerType] = (
        ["batter", "pitcher"] if player_type == "both" else [player_type]  # type: ignore[list-item]
    )
    source = SavantLeaderboards()
    db = _database()
    for ptype in types:
        df = source.fetch_expected_stats(season, ptype, min_pa=min_pa)
        written = db.upsert("savant_expected_stats", df, keys=list(SAVANT_KEYS))
        console.print(f"[green]savant_expected_stats[/green] ({ptype}): {written} filas.")
        df = source.fetch_statcast_metrics(season, ptype, min_pa=min_pa)
        written = db.upsert("savant_statcast_metrics", df, keys=list(SAVANT_KEYS))
        console.print(f"[green]savant_statcast_metrics[/green] ({ptype}): {written} filas.")


@ingest_app.command()
def lahman(
    tables: str = typer.Option(
        "people,batting,pitching,teams",
        help=f"Tablas separadas por coma. Opciones: {','.join(sorted(LAHMAN_TABLES))}.",
    ),
) -> None:
    """Descarga tablas de la Lahman Database (CRAN) -> tablas ``lahman_*``.

    Requiere: poetry install --with data
    """
    names = [t.strip() for t in tables.split(",") if t.strip()]
    unknown = [n for n in names if n not in LAHMAN_TABLES]
    if unknown:
        raise typer.BadParameter(f"Tablas no soportadas: {unknown}")
    source = LahmanSource(cache_dir=get_settings().external_dir / "lahman")
    db = _database()
    try:
        for name in names:
            spec = LAHMAN_TABLES[name]
            df = source.fetch_table(name)
            written = db.upsert(spec.db_table, df, keys=list(spec.keys))
            console.print(f"[green]{spec.db_table}:[/green] {written} filas.")
    except ImportError as exc:
        console.print("[red]pyreadr no instalado.[/red] Ejecuta: poetry install --with data")
        raise typer.Exit(code=1) from exc


@ingest_app.command()
def boxscores(
    start: str = typer.Option(..., help="Fecha inicial (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Fecha final (YYYY-MM-DD). Por defecto, igual a start."),
) -> None:
    """Descarga boxscores de juegos finalizados -> ``lineups`` y ``pitching_lines``.

    Requiere que ``games`` esté poblada para el rango (mlb ingest schedule).
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date
    db = _database()
    if not db.table_exists("games"):
        console.print("[red]Tabla games vacía.[/red] Corre antes: mlb ingest schedule")
        raise typer.Exit(code=1)

    game_pks = db.query(
        f"SELECT game_pk FROM games WHERE status = 'Final' "
        f"AND game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}' "
        f"ORDER BY game_pk"
    )["game_pk"].to_list()
    if not game_pks:
        console.print("[yellow]Sin juegos finalizados en el rango.[/yellow]")
        return

    api = MlbStatsApi()
    lineup_frames: list[pl.DataFrame] = []
    pitching_frames: list[pl.DataFrame] = []
    for game_pk in game_pks:
        lineups_df, pitching_df = api.fetch_boxscore(game_pk)
        lineup_frames.append(lineups_df)
        pitching_frames.append(pitching_df)

    written_lineups = db.upsert(
        "lineups", pl.concat(lineup_frames), keys=["game_pk", "player_id"]
    )
    written_pitching = db.upsert(
        "pitching_lines", pl.concat(pitching_frames), keys=["game_pk", "player_id"]
    )
    console.print(
        f"[green]lineups:[/green] {written_lineups} filas, "
        f"[green]pitching_lines:[/green] {written_pitching} filas "
        f"({len(game_pks)} juegos, {start_date} -> {end_date})."
    )


@ingest_app.command()
def odds(
    markets: str = typer.Option("h2h,totals", help="Mercados separados por coma."),
    regions: str = typer.Option("us", help="Regiones de casas separadas por coma."),
) -> None:
    """Descarga snapshot de cuotas de The Odds API -> tabla ``odds_snapshots``.

    Cada corrida añade un snapshot nuevo (no pisa los anteriores): el
    histórico de snapshots es la base para medir CLV.

    Requiere ``ODDS_API_KEY`` en ``.env``.
    """
    try:
        source = OddsApiSource(api_key=get_settings().odds_api_key)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    df = source.fetch_odds(
        markets=tuple(m.strip() for m in markets.split(",") if m.strip()),
        regions=tuple(r.strip() for r in regions.split(",") if r.strip()),
    )
    if df.is_empty():
        console.print("[yellow]La API no devolvió eventos (¿sin juegos próximos?).[/yellow]")
        return
    written = _database().upsert("odds_snapshots", df, keys=list(ODDS_KEYS))
    console.print(
        f"[green]odds_snapshots:[/green] {written} filas escritas "
        f"({df['event_id'].n_unique()} eventos)."
    )


@ingest_app.command()
def weather(
    start: str = typer.Option(..., help="Fecha inicial (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Fecha final (YYYY-MM-DD). Por defecto, igual a start."),
) -> None:
    """Descarga clima horario por estadio con juegos -> tabla ``weather_hourly``.

    Requiere ``games`` y ``venues`` pobladas (mlb ingest schedule / venues).
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date
    db = _database()
    prerequisites = {"games": "mlb ingest schedule", "venues": "mlb ingest venues"}
    for table, command in prerequisites.items():
        if not db.table_exists(table):
            console.print(f"[red]Tabla {table} vacía.[/red] Corre antes: {command}")
            raise typer.Exit(code=1)

    targets = db.query(
        f"SELECT DISTINCT v.venue_id, v.latitude, v.longitude "
        f"FROM games g JOIN venues v ON g.venue_id = v.venue_id "
        f"WHERE g.game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}' "
        f"AND v.latitude IS NOT NULL"
    )
    if targets.is_empty():
        console.print("[yellow]Sin estadios con juegos en el rango.[/yellow]")
        return

    source = WeatherSource()
    frames = [
        source.fetch_hourly(
            venue_id=row["venue_id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            start_date=start_date,
            end_date=end_date,
        )
        for row in targets.iter_rows(named=True)
    ]
    written = db.upsert("weather_hourly", pl.concat(frames), keys=["venue_id", "time_utc"])
    console.print(
        f"[green]weather_hourly:[/green] {written} filas "
        f"({len(frames)} estadios, {start_date} -> {end_date})."
    )
