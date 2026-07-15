"""Subcomandos ``mlb simulate``: simulación Monte Carlo de juegos."""

import logging
from datetime import date, datetime

import polars as pl
import typer
from rich.console import Console

from mlb_quant.db import Database
from mlb_quant.models.poisson import PoissonRunsModel
from mlb_quant.settings import get_settings
from mlb_quant.simulations.monte_carlo import SimulationConfig, simulate_games

logger = logging.getLogger(__name__)
console = Console()

simulate_app = typer.Typer(help="Simula juegos con Monte Carlo desde el modelo Poisson.")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"Fecha inválida: {value!r}. Formato: YYYY-MM-DD.") from exc


@simulate_app.command()
def run(
    start: str = typer.Option(..., help="Primer día a simular (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Último día. Por defecto, igual a start."),
    sims: int = typer.Option(10_000, help="Simulaciones por juego."),
    seed: int = typer.Option(42, help="Semilla (reproducibilidad)."),
    total_line: float = typer.Option(8.5, help="Línea de over/under."),
) -> None:
    """Entrena con historia previa, simula el rango -> tabla ``simulations_game``.

    El modelo se entrena SOLO con juegos anteriores a start (sin leakage).
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else start_date

    db = Database(get_settings().duckdb_path)
    if not db.table_exists("features_game"):
        console.print("[red]Tabla features_game vacía.[/red] Corre antes: mlb features build")
        raise typer.Exit(code=1)
    features = db.query("SELECT * FROM features_game")

    train = features.filter(pl.col("game_date") < start_date)
    target = features.filter(
        (pl.col("game_date") >= start_date) & (pl.col("game_date") <= end_date)
    )
    if train.height < 500:
        console.print(f"[red]Historia insuficiente antes de {start_date} ({train.height}).[/red]")
        raise typer.Exit(code=1)
    if target.is_empty():
        console.print("[yellow]Sin juegos en el rango.[/yellow]")
        return

    model = PoissonRunsModel().fit(train)
    lambdas = model.predict_lambdas(target)
    config = SimulationConfig(n_sims=sims, seed=seed, total_line=total_line)
    result = simulate_games(lambdas, config)

    written = db.upsert("simulations_game", result, keys=["game_pk"])
    console.print(
        f"[green]simulations_game:[/green] {written} juegos x {sims} sims "
        f"({start_date} -> {end_date})."
    )
