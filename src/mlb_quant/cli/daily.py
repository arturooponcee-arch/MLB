"""Comando ``mlb daily``: pipeline diario completo.

Sustituye a un orquestador (Prefect/Airflow): para un sistema de una
máquina, un comando idempotente + el programador del sistema operativo
cubren la necesidad sin servidor ni dependencias extra. Programarlo:

Windows (Task Scheduler)::

    schtasks /create /tn "MLB Quant Daily" /sc daily /st 09:00 ^
        /tr "poetry run mlb daily"
"""

import logging
from datetime import date, datetime, timedelta

import typer
from rich.console import Console

from mlb_quant.cli.ingest import boxscores, schedule, statcast, weather
from mlb_quant.db import Database
from mlb_quant.preprocessing.batter_games import build_batter_games
from mlb_quant.reporting.daily import generate_daily_report
from mlb_quant.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

daily_app = typer.Typer(help="Pipeline diario: ingesta -> features -> reportes.")


@daily_app.command("run")
def run(
    target: str = typer.Option(
        None, "--date", help="Fecha 'hoy' del pipeline (YYYY-MM-DD)."
    ),
    with_statcast: bool = typer.Option(
        True, help="Descarga Statcast de ayer (~1-2 min)."
    ),
    sims: int = typer.Option(10_000, help="Simulaciones Monte Carlo por juego."),
) -> None:
    """Corre el pipeline diario completo. Idempotente: re-correrlo no duplica.

    Pasos: schedule (ayer y próximos días) -> boxscores de ayer ->
    statcast de ayer -> batter_games -> features de ayer -> clima ->
    snapshot de cuotas (si hay ODDS_API_KEY) -> dashboard + reporte diario.
    """
    today = datetime.strptime(target, "%Y-%m-%d").date() if target else date.today()
    yesterday = today - timedelta(days=1)
    horizon = today + timedelta(days=2)
    db = Database(get_settings().duckdb_path)

    steps: list[tuple[str, object]] = [
        ("schedule", lambda: schedule(start=str(yesterday), end=str(horizon))),
        ("boxscores", lambda: boxscores(start=str(yesterday), end=str(yesterday))),
    ]
    if with_statcast:
        steps.append(("statcast", lambda: statcast(start=str(yesterday), end=str(yesterday))))
        steps.append(("batter_games", lambda: build_batter_games(db)))
    steps += [
        ("weather", lambda: weather(start=str(yesterday), end=str(horizon))),
        ("features", lambda: _build_features(str(yesterday))),
    ]
    if get_settings().odds_api_key:
        from mlb_quant.cli.ingest import odds

        # Los defaults de typer son OptionInfo: en llamadas directas hay
        # que pasar los argumentos explícitos.
        steps.append(("odds", lambda: odds(markets="h2h,totals", regions="us")))

    failures = []
    for name, step in steps:
        console.rule(f"[bold]{name}")
        try:
            step()  # type: ignore[operator]
        except typer.Exit:
            failures.append(name)
        except Exception as exc:
            logger.exception("Paso %s falló.", name)
            failures.append(f"{name}: {exc}")

    console.rule("[bold]reporte")
    written = generate_daily_report(
        db, today, get_settings().reports_dir / "daily", sims=sims
    )
    from mlb_quant.cli.report import dashboard as dashboard_command

    dashboard_command(target=str(today), watch=False, interval=0, sims=sims, refresh_data=False)

    if failures:
        console.print(f"[yellow]Pasos con fallos: {failures}[/yellow]")
    console.print(f"[green]Pipeline diario completo.[/green] {len(written)} archivos de reporte.")


def _build_features(day: str) -> None:
    from mlb_quant.cli.features import build as features_build

    features_build(start=day, end=day, blocks="")
