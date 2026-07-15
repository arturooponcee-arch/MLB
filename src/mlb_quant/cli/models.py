"""Subcomandos ``mlb models``: entrenamiento y backtesting."""

import logging
from datetime import date, datetime

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.db import Database
from mlb_quant.evaluation.backtest import walk_forward
from mlb_quant.evaluation.metrics import calibration_table
from mlb_quant.models.base import save_model
from mlb_quant.models.poisson import PoissonRunsModel
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

models_app = typer.Typer(help="Entrena modelos y corre backtests walk-forward.")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"Fecha inválida: {value!r}. Formato: YYYY-MM-DD.") from exc


def _load_features(db: Database) -> pl.DataFrame:
    if not db.table_exists("features_game"):
        console.print("[red]Tabla features_game vacía.[/red] Corre antes: mlb features build")
        raise typer.Exit(code=1)
    return db.query("SELECT * FROM features_game")


@models_app.command()
def backtest(
    start: str = typer.Option(..., help="Primer día a predecir (YYYY-MM-DD)."),
    end: str = typer.Option(..., help="Último día a predecir (YYYY-MM-DD)."),
    total_line: float = typer.Option(8.5, help="Línea de over/under a evaluar."),
) -> None:
    """Backtest walk-forward mensual -> tabla ``backtest_predictions``."""
    start_date, end_date = _parse_date(start), _parse_date(end)
    db = Database(get_settings().duckdb_path)
    features = _load_features(db)

    try:
        result = walk_forward(features, start_date, end_date, total_line=total_line)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    db.upsert("backtest_predictions", result.predictions, keys=["game_pk"])

    for name, metrics in result.metrics.items():
        table = Table(title=name)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green", justify="right")
        for key, value in metrics.items():
            table.add_row(key, f"{value:.4f}" if key != "n" else f"{int(value)}")
        console.print(table)

    home_win = (
        (result.predictions["home_score"] > result.predictions["away_score"])
        .to_numpy()
        .astype(int)
    )
    calibration = calibration_table(
        home_win, result.predictions["p_home_ml"].to_numpy()
    )
    cal_table = Table(title="Calibración moneyline (Poisson)")
    for column in ("bin", "p_predicha", "p_observada", "n"):
        cal_table.add_column(column, justify="right")
    for row in calibration.iter_rows(named=True):
        cal_table.add_row(
            f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
            f"{row['p_predicted']:.3f}",
            f"{row['p_observed']:.3f}",
            str(row["n"]),
        )
    console.print(cal_table)


@models_app.command()
def train(
    start: str = typer.Option(..., help="Primer día de entrenamiento (YYYY-MM-DD)."),
    end: str = typer.Option(..., help="Último día de entrenamiento (YYYY-MM-DD)."),
    output: str = typer.Option("poisson_runs.joblib", help="Archivo destino en models/."),
) -> None:
    """Entrena el Poisson con un rango y lo serializa en ``models/``."""
    start_date, end_date = _parse_date(start), _parse_date(end)
    db = Database(get_settings().duckdb_path)
    features = _load_features(db).filter(
        (pl.col("game_date") >= start_date) & (pl.col("game_date") <= end_date)
    )
    if features.is_empty():
        console.print("[yellow]Sin juegos en el rango.[/yellow]")
        raise typer.Exit(code=1)

    model = PoissonRunsModel().fit(features)
    path = get_settings().models_dir / output
    save_model(model, path)
    console.print(f"[green]Modelo guardado:[/green] {path} ({len(features)} juegos).")
