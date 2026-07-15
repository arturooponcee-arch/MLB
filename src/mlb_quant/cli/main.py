"""Punto de entrada de la CLI ``mlb``.

Cada fase del proyecto añade aquí sus subcomandos (``ingest``, ``features``,
``train``, ``simulate``, ``bets``, ``report``...).
"""

import typer
from rich.console import Console
from rich.table import Table

from mlb_quant import __version__
from mlb_quant.cli.features import features_app
from mlb_quant.cli.ingest import ingest_app
from mlb_quant.cli.models import models_app
from mlb_quant.cli.report import report_app
from mlb_quant.cli.simulate import simulate_app
from mlb_quant.settings import get_settings
from mlb_quant.utils.logging import setup_logging

app = typer.Typer(
    name="mlb",
    help="MLB Quant: análisis cuantitativo y pronóstico para MLB.",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(simulate_app, name="simulate")
app.add_typer(report_app, name="report")
console = Console()


@app.callback()
def main() -> None:
    """Inicializa logging antes de cualquier subcomando."""
    settings = get_settings()
    setup_logging(level=settings.log_level, logs_dir=settings.logs_dir)


@app.command()
def info() -> None:
    """Muestra versión, rutas y estado de la configuración."""
    settings = get_settings()

    table = Table(title=f"MLB Quant v{__version__}")
    table.add_column("Clave", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Raíz de datos", str(settings.data_dir))
    table.add_row("DuckDB", str(settings.duckdb_path))
    table.add_row("Logs", str(settings.logs_dir))
    table.add_row("Nivel de log", settings.log_level)
    table.add_row("Odds API key", "configurada" if settings.odds_api_key else "no configurada")

    console.print(table)


if __name__ == "__main__":
    app()
