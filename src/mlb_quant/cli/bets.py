"""Subcomandos ``mlb bets``: sizing manual con Kelly y escaneo EV+ del mercado."""

from datetime import date, datetime
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from mlb_quant.bankroll.kelly import expected_value, kelly_fraction, recommend_stake
from mlb_quant.betting.ledger import (
    BETS_KEYS,
    closing_odds,
    create_bet,
    ledger_summary,
    settle_bets,
)
from mlb_quant.betting.manual_odds import load_manual_odds, write_template
from mlb_quant.betting.value import find_moneyline_value, normalized_team
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
def template(
    target: str = typer.Option(None, "--date", help="Fecha de los juegos (YYYY-MM-DD). Hoy."),
    output: str = typer.Option(
        None, help="Ruta del CSV. Default: reports/daily/<fecha>_cuotas.csv."
    ),
) -> None:
    """Genera plantilla CSV con los juegos del día para teclear cuotas del book.

    Rellena ``odds_away`` y ``odds_home`` (decimales) y pásala a
    ``mlb bets scan --odds-file``. Varias casas: repite la fila del juego
    cambiando ``book``.
    """
    game_date = datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("games"):
        console.print("[red]Tabla games vacía.[/red] Corre antes: mlb ingest schedule")
        raise typer.Exit(code=1)
    games = db.query(
        f"SELECT game_pk, game_datetime_utc, away_team_name, home_team_name "
        f"FROM games WHERE game_date = DATE '{game_date}' ORDER BY game_datetime_utc"
    )
    if games.is_empty():
        console.print(f"[yellow]Sin juegos programados el {game_date}.[/yellow]")
        return
    path = (
        Path(output)
        if output
        else get_settings().reports_dir / "daily" / f"{game_date}_cuotas.csv"
    )
    n_games = write_template(path, games)
    console.print(
        f"[green]Plantilla escrita:[/green] {path} ({n_games} juegos). "
        f"Rellena las cuotas y corre: mlb bets scan --odds-file {path}"
    )


@bets_app.command()
def scan(
    target: str = typer.Option(None, "--date", help="Fecha de los juegos (YYYY-MM-DD). Hoy."),
    odds_file: str = typer.Option(
        None, "--odds-file", help="CSV de cuotas manuales (mlb bets template). Sin API."
    ),
    bankroll: float = typer.Option(1000.0, help="Bankroll actual."),
    kelly_mult: float = typer.Option(0.5, "--kelly", help="Fracción de Kelly (0.5 = half)."),
    cap: float = typer.Option(0.05, help="Tope duro (fracción del bankroll)."),
    min_ev: float = typer.Option(0.02, help="EV mínimo por unidad para reportar."),
    sims: int = typer.Option(10_000, help="Simulaciones Monte Carlo por juego."),
) -> None:
    """Escanea el mercado: cuotas vs. modelo -> apuestas moneyline EV+.

    Las cuotas vienen de un CSV manual (``--odds-file``, sin API) o de
    The Odds API (requiere ``ODDS_API_KEY``). En ambos casos el snapshot
    se persiste en ``odds_snapshots``: el histórico del mercado alimenta
    el análisis de CLV y el contraste modelo vs. mercado.
    """
    game_date = datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    settings = get_settings()
    db = Database(settings.duckdb_path)

    games = predict_upcoming_games(db, game_date, sims)
    if games.is_empty():
        console.print(f"[yellow]Sin juegos programados el {game_date}.[/yellow]")
        return

    odds = _load_odds(odds_file, games)
    if odds.is_empty():
        console.print("[yellow]Sin cuotas (¿API sin eventos?).[/yellow]")
        return
    db.upsert("odds_snapshots", odds, keys=list(ODDS_KEYS))

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


@bets_app.command()
def log(
    game: int = typer.Option(None, help="game_pk del juego (ver mlb bets template)."),
    target: str = typer.Option(None, "--date", help="Alternativa: fecha (YYYY-MM-DD)."),
    team: str = typer.Option(None, help="Alternativa: equipo apostado (con --date)."),
    side: str = typer.Option(None, help="home | away (obligatorio con --game)."),
    odds: float = typer.Option(..., help="Cuota decimal tomada."),
    stake: float = typer.Option(..., help="Monto apostado."),
    book: str = typer.Option("manual", help="Casa de apuestas."),
    p_model: float = typer.Option(None, help="Probabilidad del modelo al apostar."),
) -> None:
    """Registra una apuesta moneyline en el ledger (tabla ``bets``).

    Identifica el juego por ``--game <pk>`` + ``--side``, o por
    ``--date`` + ``--team`` (el lado se infiere del nombre). El ledger
    alimenta ``mlb bets settle`` (resultados + CLV) y ``mlb bets report``.
    """
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("games"):
        console.print("[red]Tabla games vacía.[/red] Corre antes: mlb ingest schedule")
        raise typer.Exit(code=1)

    row, selection = _resolve_game(db, game, target, team, side)
    if row["status"] == "Final":
        console.print("[red]El juego ya terminó: no se registran apuestas post-juego.[/red]")
        raise typer.Exit(code=1)
    try:
        bet = create_bet(row, selection, odds, stake, book, p_model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    db.upsert("bets", bet, keys=list(BETS_KEYS))
    record = bet.row(0, named=True)
    console.print(
        f"[green]Apuesta registrada[/green] {record['bet_id']}: "
        f"{record['team_name']} ML @ {odds:.2f} x {stake:.2f} ({book})."
    )


@bets_app.command()
def settle() -> None:
    """Liquida apuestas abiertas con juegos finalizados y calcula su CLV."""
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("bets"):
        console.print("[yellow]Sin apuestas registradas (mlb bets log).[/yellow]")
        return
    bets = db.query("SELECT * FROM bets")
    games = db.query("SELECT * FROM games")
    settled = settle_bets(bets, games)
    if settled.is_empty():
        console.print("[yellow]Nada que liquidar (sin juegos finalizados nuevos).[/yellow]")
        return

    if db.table_exists("odds_snapshots"):
        clv = closing_odds(settled, db.query("SELECT * FROM odds_snapshots"), games)
        settled = settled.drop("closing_odds", "clv_pct").join(clv, on="bet_id", how="left")
    db.upsert("bets", settled, keys=list(BETS_KEYS))

    table = Table(title="Apuestas liquidadas")
    for column in ("Equipo", "Cuota", "Stake", "Resultado", "Profit", "Cierre", "CLV"):
        table.add_column(column, justify="right")
    for row in settled.iter_rows(named=True):
        color = {"won": "green", "lost": "red"}.get(row["status"], "yellow")
        table.add_row(
            row["team_name"],
            f"{row['decimal_odds']:.2f}",
            f"{row['stake']:.2f}",
            f"[{color}]{row['status']}[/{color}]",
            f"{row['profit']:+.2f}",
            f"{row['closing_odds']:.2f}" if row["closing_odds"] else "-",
            f"{row['clv_pct']:+.2%}" if row["clv_pct"] is not None else "-",
        )
    console.print(table)


@bets_app.command()
def report(
    season: int = typer.Option(None, help="Filtra por temporada (año de game_date)."),
) -> None:
    """Resumen del ledger: resultados, ROI y CLV medio."""
    db = Database(get_settings().duckdb_path)
    if not db.table_exists("bets"):
        console.print("[yellow]Sin apuestas registradas (mlb bets log).[/yellow]")
        return
    bets = db.query("SELECT * FROM bets ORDER BY game_date, placed_at_utc")
    if season:
        bets = bets.filter(pl.col("game_date").dt.year() == season)
    if bets.is_empty():
        console.print("[yellow]Sin apuestas en el filtro.[/yellow]")
        return

    table = Table(title=f"Ledger de apuestas ({len(bets)})")
    for column in ("Fecha", "Equipo", "Cuota", "Stake", "Estado", "Profit", "CLV"):
        table.add_column(column, justify="right")
    for row in bets.iter_rows(named=True):
        color = {"won": "green", "lost": "red", "open": "cyan"}.get(row["status"], "yellow")
        table.add_row(
            str(row["game_date"]),
            row["team_name"],
            f"{row['decimal_odds']:.2f}",
            f"{row['stake']:.2f}",
            f"[{color}]{row['status']}[/{color}]",
            f"{row['profit']:+.2f}" if row["profit"] is not None else "-",
            f"{row['clv_pct']:+.2%}" if row["clv_pct"] is not None else "-",
        )
    console.print(table)

    summary = ledger_summary(bets)
    console.print(
        f"Liquidado: {summary['staked']:.2f} | Profit: {summary['profit']:+.2f} "
        f"| ROI: {summary['roi']:+.2%} | Aciertos: {summary['hit_rate']:.1%} "
        f"| CLV medio: {summary['clv_mean']:+.2%} ({int(summary['n_clv'])} con cierre)"
    )
    if summary["n_clv"] and summary["clv_mean"] < 0:
        console.print(
            "[yellow]CLV medio negativo: el mercado cierra mejor que tus cuotas — "
            "revisa timing y books antes de confiar en el ROI.[/yellow]"
        )


def _resolve_game(
    db: Database, game: int | None, target: str | None, team: str | None, side: str | None
) -> tuple[dict[str, object], str]:
    """Resuelve (fila de games, selection) desde --game/--side o --date/--team."""
    if game is not None:
        if side not in ("home", "away"):
            console.print("[red]Con --game indica --side home|away.[/red]")
            raise typer.Exit(code=1)
        rows = db.query(f"SELECT * FROM games WHERE game_pk = {int(game)}")
        if rows.is_empty():
            console.print(f"[red]game_pk {game} no existe en games.[/red]")
            raise typer.Exit(code=1)
        return rows.row(0, named=True), side

    if not target or not team:
        console.print("[red]Indica --game <pk> --side, o --date + --team.[/red]")
        raise typer.Exit(code=1)
    game_date = datetime.strptime(target, "%Y-%m-%d").date()
    day = db.query(f"SELECT * FROM games WHERE game_date = DATE '{game_date}'")
    needle = team.lower().strip()
    matches = day.filter(
        normalized_team("home_team_name").str.contains(needle, literal=True)
        | normalized_team("away_team_name").str.contains(needle, literal=True)
    )
    if len(matches) != 1:
        names = sorted(
            set(day["home_team_name"].to_list() + day["away_team_name"].to_list())
        )
        console.print(
            f"[red]{len(matches)} juegos casan con {team!r} el {game_date}.[/red] "
            f"Equipos del día: {names}. Usa --game <pk> si hay doubleheader."
        )
        raise typer.Exit(code=1)
    row = matches.row(0, named=True)
    selection = "home" if needle in row["home_team_name"].lower() else "away"
    return row, selection


def _load_odds(odds_file: str | None, games: pl.DataFrame) -> pl.DataFrame:
    """Carga cuotas del CSV manual o de The Odds API, con errores legibles."""
    if odds_file:
        try:
            return load_manual_odds(Path(odds_file), games)
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    try:
        source = OddsApiSource(api_key=get_settings().odds_api_key)
    except ValueError as exc:
        console.print(
            f"[red]{exc}[/red]\n"
            "[dim]Sin API: mlb bets template genera un CSV para teclear cuotas "
            "y pasarlo con --odds-file.[/dim]"
        )
        raise typer.Exit(code=1) from exc
    return source.fetch_odds()
