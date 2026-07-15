"""Subcomandos ``mlb bets``: sizing manual con Kelly."""

import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.bankroll.kelly import expected_value, kelly_fraction, recommend_stake

console = Console()

bets_app = typer.Typer(help="EV y Kelly para apuestas manuales (tú traes la cuota).")


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
