"""Subcomandos ``mlb bets``: sizing manual con Kelly y escaneo EV+ del mercado."""

from datetime import date, datetime

import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.bankroll.kelly import expected_value, kelly_fraction, recommend_stake
from mlb_quant.betting.value import find_moneyline_value
from mlb_quant.db import Database
from mlb_quant.ingestion.odds_api import ODDS_KEYS, OddsApiSource
from mlb_quant.reporting.daily import predict_upcoming_games
from mlb_quant.settings import get_settings

console = Console()

bets_app = typer.Typer(help="EV y Kelly: sizing manual o escaneo automático del mercado.")


@bets_app.command()
def kelly(
    p: float = typer.Option(..., help="Probabilidad del modelo (0-1)."),
    odds: float = typer.Option(..., help="Cuota decimal del book."),
    bankroll: float = typer.Option(1000.0, help="Bankroll actual."),
    cap: float = typer.Option(0.05, help="Tope duro (fracción del bankroll)."),
) -> None:
    """EV, cuota justa y stakes Kelly (full/half/quarter) para una apuesta."""
    try:
        ev = expected_value(p, odds)
        full = kelly_fraction(p, odds)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Análisis de apuesta")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green", justify="right")
    table.add_row("Probabilidad modelo", f"{p:.3f}")
    table.add_row("Cuota del book", f"{odds:.2f}")
    table.add_row("Cuota justa (umbral)", f"{1 / p:.2f}")
    table.add_row("EV por unidad", f"{ev:+.4f}")
    table.add_row("Kelly completo", f"{full:.4f}")
    for label, multiplier in (("Half Kelly", 0.5), ("Quarter Kelly", 0.25)):
        rec = recommend_stake(bankroll, p, odds, kelly_multiplier=multiplier, max_fraction=cap)
        suffix = " (tope aplicado)" if rec.capped else ""
        table.add_row(f"Stake {label}", f"{rec.stake:.2f}{suffix}")
    console.print(table)

    if ev <= 0:
        console.print(
            "[yellow]Sin ventaja: la cuota no supera la cuota justa. No apostar.[/yellow]"
        )


@bets_app.command()
def scan(
    target: str = typer.Option(None, "--date", help="Fecha de los juegos (YYYY-MM-DD). Hoy."),
    bankroll: float = typer.Option(1000.0, help="Bankroll actual."),
    kelly_mult: float = typer.Option(0.5, "--kelly", help="Fracción de Kelly (0.5 = half)."),
    cap: float = typer.Option(0.05, help="Tope duro (fracción del bankroll)."),
    min_ev: float = typer.Option(0.02, help="EV mínimo por unidad para reportar."),
    sims: int = typer.Option(10_000, help="Simulaciones Monte Carlo por juego."),
) -> None:
    """Escanea el mercado: cuotas en vivo vs. modelo -> apuestas moneyline EV+.

    Descarga un snapshot de The Odds API (se persiste en ``odds_snapshots``
    para el histórico de CLV), predice los juegos de la fecha y reporta
    los lados cuyo mejor precio supera la cuota justa del modelo.

    Requiere ``ODDS_API_KEY`` en ``.env`` y el warehouse con features.
    """
    game_date = datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    settings = get_settings()
    try:
        source = OddsApiSource(api_key=settings.odds_api_key)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    odds = source.fetch_odds()
    if odds.is_empty():
        console.print("[yellow]La API no devolvió cuotas (¿sin juegos próximos?).[/yellow]")
        return
    db = Database(settings.duckdb_path)
    db.upsert("odds_snapshots", odds, keys=list(ODDS_KEYS))

    games = predict_upcoming_games(db, game_date, sims)
    if games.is_empty():
        console.print(f"[yellow]Sin juegos programados el {game_date}.[/yellow]")
        return

    value = find_moneyline_value(
        games,
        odds,
        bankroll=bankroll,
        kelly_multiplier=kelly_mult,
        max_fraction=cap,
        min_ev=min_ev,
    )
    if value.is_empty():
        console.print(
            f"[yellow]Sin valor: ningún lado supera EV {min_ev:+.2%} "
            f"({len(games)} juegos evaluados).[/yellow]"
        )
        return

    table = Table(title=f"Apuestas EV+ — {game_date} (bankroll {bankroll:,.0f})")
    table.add_column("Juego", style="cyan")
    table.add_column("Apostar a", style="bold")
    table.add_column("p modelo", justify="right")
    table.add_column("Justa", justify="right")
    table.add_column("Mejor cuota", justify="right", style="green")
    table.add_column("Casa")
    table.add_column("EV", justify="right", style="green")
    table.add_column("Stake", justify="right")
    for row in value.iter_rows(named=True):
        suffix = " (tope)" if row["capped"] else ""
        table.add_row(
            f"{row['away_team_name']} @ {row['home_team_name']}",
            row["team"],
            f"{row['p_model']:.3f}",
            f"{row['fair_odds']:.2f}",
            f"{row['best_odds']:.2f}",
            row["book"],
            f"{row['ev']:+.2%}",
            f"{row['kelly_stake']:.2f}{suffix}",
        )
    console.print(table)
    console.print(
        f"[dim]{len(value)} apuesta(s) con EV > {min_ev:+.2%} de "
        f"{len(games)} juegos. Kelly x{kelly_mult}, tope {cap:.0%}.[/dim]"
    )
