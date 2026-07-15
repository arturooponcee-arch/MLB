"""Subcomandos ``mlb props``: props por jugador."""

import logging
from datetime import date, datetime, timedelta

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.db import Database
from mlb_quant.feature_engineering.upcoming import build_upcoming_features
from mlb_quant.models.props import (
    BATTER_STATS,
    PropModel,
    build_batter_frame,
    build_pitcher_k_frame,
)
from mlb_quant.preprocessing.batter_games import build_batter_games
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

props_app = typer.Typer(help="Modelos de props por jugador (K, hits, TB, HR, BB).")

#: Líneas típicas por mercado para la evaluación.
DEFAULT_LINES: dict[str, float] = {
    "strikeouts": 5.5,
    "hits": 0.5,
    "total_bases": 1.5,
    "home_runs": 0.5,
    "walks": 0.5,
}


def _database() -> Database:
    return Database(get_settings().duckdb_path)


@props_app.command("build-batters")
def build_batters() -> None:
    """Agrega Statcast a línea por bateador-juego -> tabla ``batter_games``."""
    try:
        written = build_batter_games(_database())
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]batter_games:[/green] {written} filas.")


@props_app.command()
def evaluate(
    split: str = typer.Option("2026-01-01", help="Entrena antes de esta fecha, evalúa después."),
) -> None:
    """Evalúa cada mercado de props con corte temporal (MAE y Brier)."""
    split_date = datetime.strptime(split, "%Y-%m-%d").date()
    db = _database()

    results = Table(title=f"Props: entrenado < {split_date}, evaluado >= {split_date}")
    for column in ("mercado", "línea", "n", "MAE", "Brier over", "base rate"):
        results.add_column(column, justify="right")

    frames = {"strikeouts": build_pitcher_k_frame(db)}
    for stat in BATTER_STATS:
        frames[stat] = build_batter_frame(db, stat)

    for market, frame in frames.items():
        train = frame.filter(pl.col("game_date") < split_date)
        test = frame.filter(pl.col("game_date") >= split_date)
        if train.is_empty() or test.is_empty():
            continue
        model = PropModel().fit(train)
        lam = model.predict_lambda(test)["lambda"].to_numpy()
        actual = test["target"].cast(pl.Float64).to_numpy()
        line = DEFAULT_LINES[market]
        p_over = PropModel.probability_over(lam, line)
        over = (actual > line).astype(float)
        mae = float(abs(lam - actual).mean())
        brier = float(((p_over - over) ** 2).mean())
        results.add_row(
            market,
            str(line),
            str(len(test)),
            f"{mae:.3f}",
            f"{brier:.4f}",
            f"{over.mean():.3f}",
        )
    console.print(results)


@props_app.command()
def strikeouts(
    target: str = typer.Option(None, "--date", help="Fecha (YYYY-MM-DD). Por defecto, hoy."),
    line: float = typer.Option(5.5, help="Línea de strikeouts."),
) -> None:
    """Predice K de los abridores probables -> tabla ``props_pitcher_k``."""
    target_date = (
        datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    )
    db = _database()

    frame = build_pitcher_k_frame(db)
    model = PropModel().fit(frame.filter(pl.col("game_date") < target_date))

    upcoming = _upcoming_pitcher_frame(db, target_date)
    if upcoming.is_empty():
        console.print("[yellow]Sin juegos con abridores probables en la fecha.[/yellow]")
        return
    lam = model.predict_lambda(upcoming)
    merged = upcoming.join(lam, on=["game_pk", "player_id"]).with_columns(
        pl.lit(line).alias("line"),
        pl.Series(
            "p_over", PropModel.probability_over(lam["lambda"].to_numpy(), line)
        ),
    )
    db.upsert("props_pitcher_k", merged, keys=["game_pk", "player_id"])

    table = Table(title=f"Strikeouts del abridor — línea {line} — {target_date}")
    columns = ("Pitcher", "λ K", "P(over)", "Justa over", "P(under)", "Justa under")
    for column in columns:
        table.add_column(column, justify="right")
    for row in merged.sort("p_over", descending=True).iter_rows(named=True):
        p_over = row["p_over"]
        table.add_row(
            str(row["pitcher_name"]),
            f"{row['lambda']:.2f}",
            f"{p_over:.3f}",
            f"{1 / p_over:.2f}" if p_over > 0.001 else "—",
            f"{1 - p_over:.3f}",
            f"{1 / (1 - p_over):.2f}" if p_over < 0.999 else "—",
        )
    console.print(table)


def _upcoming_pitcher_frame(db: Database, target_date: date) -> pl.DataFrame:
    """Filas por abridor probable con las features del frame de K."""
    upcoming = build_upcoming_features(db, target_date, target_date + timedelta(days=1))
    if upcoming.is_empty():
        return upcoming
    park_cols = [c for c in upcoming.columns if c.startswith("park_")]
    sides = []
    for side, opponent in (("home", "away"), ("away", "home")):
        own = [c for c in upcoming.columns if c.startswith((f"{side}_sp_", f"{side}_spq_"))]
        sides.append(
            upcoming.filter(pl.col(f"{side}_probable_pitcher_id").is_not_null()).select(
                "game_pk",
                "game_date",
                "season",
                pl.col(f"{side}_team_id").alias("team_id"),
                pl.col(f"{side}_probable_pitcher_id").alias("player_id"),
                pl.col(f"{side}_probable_pitcher_name").alias("pitcher_name"),
                pl.col(f"{opponent}_team_name").alias("opponent_name"),
                *[pl.col(c).alias(c.removeprefix(f"{side}_")) for c in own],
                *park_cols,
            )
        )
    return pl.concat(sides)
