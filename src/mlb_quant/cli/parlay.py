"""Subcomandos ``mlb parlay``: combinadas del cerebro pronosticador."""

import logging
from datetime import date, datetime, timedelta

import polars as pl
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mlb_quant.analysts.brain import generate_daily_parlays
from mlb_quant.db import Database
from mlb_quant.models.props import predict_upcoming_strikeouts
from mlb_quant.reporting.daily import predict_upcoming_games
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

parlay_app = typer.Typer(help="Combinadas sugeridas: analistas + cerebro pronosticador.")

_SEARCH_DAYS = 7


@parlay_app.command()
def suggest(
    target: str = typer.Option(
        None, "--date", help="Fecha objetivo (YYYY-MM-DD). Por defecto, hoy."
    ),
    sims: int = typer.Option(10_000, help="Simulaciones Monte Carlo por juego."),
    with_props: bool = typer.Option(True, help="Incluye props de K como piernas."),
) -> None:
    """Genera las combinadas del día (conservadora / equilibrada / agresiva)."""
    target_date = (
        datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    )
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("features_game"):
        console.print("[red]Tabla features_game vacía.[/red] Corre antes: mlb features build")
        raise typer.Exit(code=1)

    game_date = _next_game_date(db, target_date)
    if game_date is None:
        console.print(f"[yellow]Sin juegos próximos desde {target_date}.[/yellow]")
        return

    games = predict_upcoming_games(db, game_date, sims)
    props = pl.DataFrame()
    if with_props:
        try:
            props = predict_upcoming_strikeouts(db, game_date)
        except Exception as exc:
            logger.warning("Props omitidos: %s", exc)
    parlays = generate_daily_parlays(games, props)

    if not parlays.parlays:
        console.print("[yellow]Ninguna combinada superó los umbrales hoy.[/yellow]")
        return

    for parlay in parlays.parlays:
        table = Table(
            title=(
                f"{parlay.name.capitalize()} — p {parlay.p_combined:.3f} · "
                f"cuota justa {parlay.fair_odds:.2f}"
            )
        )
        table.add_column("Pierna", style="cyan")
        table.add_column("Mercado", style="magenta")
        table.add_column("p", justify="right", style="green")
        for leg in parlay.legs.iter_rows(named=True):
            table.add_row(leg["selection"], leg["market"], f"{leg['p']:.3f}")
        console.print(table)

    if not parlays.notes.is_empty():
        console.print("[bold]Notas de analistas:[/bold]")
        for note in parlays.notes.iter_rows(named=True):
            where = f" — {note['matchup']}" if note.get("matchup") else ""
            console.print(escape(f"  [{note['analyst']}] {note['note']}{where}"))
    console.print(
        "[dim]La cuota justa es el umbral: la combinada solo tiene valor "
        "si el book paga más.[/dim]"
    )


def _next_game_date(db: Database, target: date) -> date | None:
    """Primer día desde ``target`` con juegos programados (ventana de 7 días)."""
    result = db.query(
        f"SELECT MIN(game_date) AS d FROM games "
        f"WHERE game_type = 'R' AND status != 'Final' "
        f"AND game_date BETWEEN DATE '{target}' AND DATE '{target + timedelta(days=_SEARCH_DAYS)}'"
    )
    found = result["d"][0]
    return found if isinstance(found, date) else None
