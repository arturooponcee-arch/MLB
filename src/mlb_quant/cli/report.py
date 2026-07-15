"""Subcomandos ``mlb report``: dashboard HTML de juegos y probabilidades."""

import logging
import time
from datetime import date, datetime, timedelta

import polars as pl
import typer
from rich.console import Console

from mlb_quant.db import Database
from mlb_quant.feature_engineering.upcoming import build_upcoming_features
from mlb_quant.ingestion.mlb_stats_api import MlbStatsApi
from mlb_quant.ingestion.weather import WeatherSource
from mlb_quant.models.ensemble import CalibratedEnsembleWinModel
from mlb_quant.models.poisson import PoissonRunsModel
from mlb_quant.settings import get_settings
from mlb_quant.simulations.monte_carlo import SimulationConfig, simulate_games
from mlb_quant.visualization.dashboard import render_dashboard

logger = logging.getLogger(__name__)
console = Console()

report_app = typer.Typer(help="Genera reportes y el dashboard HTML.")

#: Días hacia adelante para buscar el próximo día con juegos.
_SEARCH_DAYS = 7


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"Fecha inválida: {value!r}. Formato: YYYY-MM-DD.") from exc


def _refresh_data(db: Database, start: date, end: date) -> None:
    """Actualiza calendario (probables, estados) y clima forecast del rango."""
    api = MlbStatsApi()
    schedule = api.fetch_schedule(start, end)
    if not schedule.is_empty():
        db.upsert("games", schedule, keys=["game_pk"])
    if not db.table_exists("venues"):
        return
    targets = db.query(
        f"SELECT DISTINCT v.venue_id, v.latitude, v.longitude "
        f"FROM games g JOIN venues v ON g.venue_id = v.venue_id "
        f"WHERE g.game_date BETWEEN DATE '{start}' AND DATE '{end}' "
        f"AND v.latitude IS NOT NULL"
    )
    if targets.is_empty():
        return
    weather = WeatherSource()
    frames = [
        weather.fetch_hourly(
            venue_id=row["venue_id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            start_date=start,
            end_date=end,
        )
        for row in targets.iter_rows(named=True)
    ]
    db.upsert("weather_hourly", pl.concat(frames), keys=["venue_id", "time_utc"])


def _next_game_date(db: Database, target: date) -> date | None:
    """Primer día desde ``target`` con juegos programados (ventana de 7 días)."""
    result = db.query(
        f"SELECT MIN(game_date) AS d FROM games "
        f"WHERE game_type = 'R' AND status != 'Final' "
        f"AND game_date BETWEEN DATE '{target}' AND DATE '{target + timedelta(days=_SEARCH_DAYS)}'"
    )
    found = result["d"][0]
    return found if isinstance(found, date) else None


def _generate(db: Database, target: date, sims: int) -> tuple[date | None, int]:
    """Un ciclo completo: features -> modelos -> simulación -> HTML."""
    game_date = _next_game_date(db, target)
    output = get_settings().reports_dir / "dashboard.html"
    if game_date is None:
        render_dashboard(pl.DataFrame(), str(target), output)
        return None, 0

    upcoming = build_upcoming_features(db, game_date, game_date)
    training = db.query("SELECT * FROM features_game")

    ensemble = CalibratedEnsembleWinModel().fit(training)
    poisson = PoissonRunsModel().fit(training)

    win = ensemble.predict_home_win(upcoming)
    lambdas = poisson.predict_lambdas(upcoming)
    sims_df = simulate_games(lambdas, SimulationConfig(n_sims=sims))

    games = (
        upcoming.select(
            "game_pk",
            "game_datetime_utc",
            "home_team_name",
            "away_team_name",
            "home_probable_pitcher_name",
            "away_probable_pitcher_name",
        )
        .join(win, on="game_pk")
        .join(sims_df, on="game_pk")
        .sort("game_datetime_utc")
    )
    render_dashboard(games, str(game_date), output)
    return game_date, len(games)


@report_app.command()
def dashboard(
    target: str = typer.Option(
        None, "--date", help="Fecha objetivo (YYYY-MM-DD). Por defecto, hoy."
    ),
    watch: bool = typer.Option(False, help="Regenera en bucle (Ctrl+C para salir)."),
    interval: int = typer.Option(300, help="Segundos entre regeneraciones con --watch."),
    sims: int = typer.Option(10_000, help="Simulaciones Monte Carlo por juego."),
    refresh_data: bool = typer.Option(
        True, help="Actualiza calendario y clima antes de generar."
    ),
) -> None:
    """Genera ``reports/dashboard.html`` con juegos, probabilidades y cuotas justas.

    Con --watch se regenera cada ``interval`` segundos; el HTML además se
    auto-recarga en el navegador.
    """
    target_date = _parse_date(target) if target else date.today()
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("features_game"):
        console.print("[red]Tabla features_game vacía.[/red] Corre antes: mlb features build")
        raise typer.Exit(code=1)

    while True:
        if refresh_data:
            try:
                _refresh_data(db, target_date, target_date + timedelta(days=_SEARCH_DAYS))
            except Exception as exc:  # red caída no debe tumbar el watch
                logger.warning("No se pudo refrescar datos: %s", exc)
        game_date, n = _generate(db, target_date, sims)
        output = get_settings().reports_dir / "dashboard.html"
        if game_date is None:
            console.print(f"[yellow]Sin juegos próximos desde {target_date}.[/yellow] -> {output}")
        else:
            console.print(f"[green]Dashboard:[/green] {n} juegos del {game_date} -> {output}")
        if not watch:
            break
        time.sleep(interval)
