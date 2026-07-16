"""Subcomandos ``mlb models``: entrenamiento y backtesting."""

import logging
from datetime import date, datetime

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.db import Database
from mlb_quant.evaluation.backtest import ENSEMBLE_CONFIGS, walk_forward
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
    compare: bool = typer.Option(
        True, help="Compara las cabezas del ensemble (columnas p_home_ens_*)."
    ),
) -> None:
    """Backtest walk-forward mensual -> tabla ``backtest_predictions``."""
    start_date, end_date = _parse_date(start), _parse_date(end)
    db = Database(get_settings().duckdb_path)
    features = _load_features(db)

    try:
        result = walk_forward(
            features,
            start_date,
            end_date,
            total_line=total_line,
            ensemble_configs=ENSEMBLE_CONFIGS if compare else None,
        )
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
def calibration(
    column: str = typer.Option("p_home_ensemble", help="Columna de probabilidad a evaluar."),
    bins: int = typer.Option(10, help="Número de bins de la tabla de calibración."),
    html: str = typer.Option(None, help="Ruta de salida HTML con curvas por temporada."),
) -> None:
    """Calibración por temporada desde ``backtest_predictions`` (sin refit).

    Imprime la tabla de calibración y Brier/log-loss de cada temporada;
    con ``--html`` escribe además las curvas (SVG inline) a un archivo.
    """
    from pathlib import Path

    from mlb_quant.evaluation.metrics import probability_metrics
    from mlb_quant.visualization.calibration import render_calibration_html

    db = Database(get_settings().duckdb_path)
    if not db.table_exists("backtest_predictions"):
        console.print(
            "[red]Tabla backtest_predictions vacía.[/red] Corre antes: mlb models backtest"
        )
        raise typer.Exit(code=1)
    predictions = db.query("SELECT * FROM backtest_predictions")
    if column not in predictions.columns:
        available = sorted(c for c in predictions.columns if c.startswith("p_home"))
        console.print(f"[red]Columna {column!r} no existe.[/red] Disponibles: {available}")
        raise typer.Exit(code=1)
    predictions = predictions.filter(pl.col(column).is_not_null())

    per_season: dict[int, pl.DataFrame] = {}
    season_metrics: dict[int, dict[str, float]] = {}
    for (season,), group in predictions.group_by(
        pl.col("game_date").dt.year(), maintain_order=True
    ):
        y = (group["home_score"] > group["away_score"]).to_numpy().astype(int)
        p = group[column].to_numpy()
        per_season[int(season)] = calibration_table(y, p, n_bins=bins)
        season_metrics[int(season)] = probability_metrics(y, p)

        table = Table(title=f"Calibración {column} — temporada {season}")
        for header in ("bin", "p_predicha", "p_observada", "n"):
            table.add_column(header, justify="right")
        for row in per_season[int(season)].iter_rows(named=True):
            table.add_row(
                f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
                f"{row['p_predicted']:.3f}",
                f"{row['p_observed']:.3f}",
                str(row["n"]),
            )
        console.print(table)
        stats = season_metrics[int(season)]
        console.print(
            f"[dim]brier={stats['brier']:.4f} log_loss={stats['log_loss']:.4f} "
            f"n={int(stats['n'])}[/dim]"
        )

    if html:
        path = Path(html)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_calibration_html(per_season, column, season_metrics), encoding="utf-8"
        )
        console.print(f"[green]Curvas escritas:[/green] {path}")


@models_app.command()
def explain(
    top: int = typer.Option(15, help="Número de features a mostrar por dirección."),
    side: str = typer.Option("home", help="home | away (lambda a explicar)."),
) -> None:
    """Coeficientes estandarizados del Poisson: qué mueve las carreras.

    Features escaladas antes de la regresión: los coeficientes son
    comparables entre sí (efecto por desviación estándar).
    """
    if side not in ("home", "away"):
        raise typer.BadParameter("side debe ser home o away.")
    db = Database(get_settings().duckdb_path)
    features = _load_features(db)
    model = PoissonRunsModel().fit(features)
    ranked = model.coefficients(side)

    table = Table(title=f"Poisson {side}: efecto en carreras (por desviación estándar)")
    table.add_column("Feature", style="cyan")
    table.add_column("Coeficiente", justify="right")
    shown = pl.concat([ranked.head(top), ranked.tail(top)]).unique(
        subset=["feature"], keep="first", maintain_order=True
    )
    for row in shown.iter_rows(named=True):
        value = row["coefficient"]
        color = "green" if value > 0 else "red"
        table.add_row(row["feature"], f"[{color}]{value:+.4f}[/{color}]")
    console.print(table)


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
