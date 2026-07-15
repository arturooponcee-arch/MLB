"""Tests de la CLI."""

from typer.testing import CliRunner

from mlb_quant import __version__
from mlb_quant.cli.main import app

runner = CliRunner()


def test_info_exits_zero() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "mlb" in result.output.lower()
