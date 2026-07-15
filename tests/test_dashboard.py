"""Tests del generador de dashboard HTML."""

from pathlib import Path

import polars as pl

from mlb_quant.visualization.dashboard import render_dashboard


def _games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1],
            "game_datetime_utc": ["2026-07-17T23:05:00Z"],
            "home_team_name": ["New York Yankees"],
            "away_team_name": ["Boston Red Sox"],
            "home_probable_pitcher_name": ["Gerrit Cole"],
            "away_probable_pitcher_name": [None],
            "p_home_win": [0.58],
            "sim_p_over": [0.55],
            "sim_p_home_runline": [0.40],
            "sim_p_f5_home_ml": [0.45],
            "sim_exp_total": [9.2],
        }
    )


def test_renders_games_and_fair_odds(tmp_path: Path) -> None:
    output = render_dashboard(_games(), "2026-07-17", tmp_path / "d.html")
    content = output.read_text(encoding="utf-8")
    assert "New York Yankees" in content
    assert "58.0%" in content
    assert f"{1 / 0.58:.2f}" in content  # cuota justa del local
    assert "por anunciar" in content  # probable ausente
    assert 'http-equiv="refresh"' in content


def test_renders_empty_state(tmp_path: Path) -> None:
    output = render_dashboard(pl.DataFrame(), "2026-07-15", tmp_path / "d.html")
    content = output.read_text(encoding="utf-8")
    assert "Sin juegos programados" in content


def test_escapes_html(tmp_path: Path) -> None:
    games = _games().with_columns(
        pl.lit("<script>alert(1)</script>").alias("home_team_name")
    )
    content = render_dashboard(games, "2026-07-17", tmp_path / "d.html").read_text(
        encoding="utf-8"
    )
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
