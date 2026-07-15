"""Tests del ensamblado de la matriz de features."""

from datetime import date

import polars as pl
import pytest

import mlb_quant.feature_engineering.blocks  # noqa: F401  (registra los bloques)
from mlb_quant.db import Database
from mlb_quant.feature_engineering.builder import build_game_features


def test_builds_one_row_per_game(warehouse: Database) -> None:
    df = build_game_features(warehouse, date(2024, 6, 1), date(2024, 6, 3))
    assert df.height == 3
    assert df["game_pk"].is_duplicated().sum() == 0


def test_team_blocks_join_both_sides(warehouse: Database) -> None:
    df = build_game_features(warehouse, date(2024, 6, 1), date(2024, 6, 3))
    assert "home_team_win_rate_5" in df.columns
    assert "away_team_win_rate_5" in df.columns
    assert "home_sp_fip_3" in df.columns
    assert "park_run_factor" in df.columns
    assert "wx_wind_out" in df.columns


def test_labels_present(warehouse: Database) -> None:
    df = build_game_features(warehouse, date(2024, 6, 1), date(2024, 6, 3))
    game3 = df.filter(pl.col("game_pk") == 3)
    assert game3["home_score"][0] == 6
    assert game3["home_team_win_rate_5"][0] == 1.0


def test_missing_table_raises(tmp_path_factory: pytest.TempPathFactory) -> None:
    empty = Database(tmp_path_factory.mktemp("empty") / "e.duckdb")
    with pytest.raises(ValueError, match="Faltan tablas"):
        build_game_features(empty, date(2024, 6, 1), date(2024, 6, 1))


def test_subset_of_blocks(warehouse: Database) -> None:
    df = build_game_features(
        warehouse, date(2024, 6, 1), date(2024, 6, 3), block_names=["team_form"]
    )
    assert "home_team_win_rate_5" in df.columns
    assert "park_run_factor" not in df.columns


def test_unknown_block_raises(warehouse: Database) -> None:
    with pytest.raises(ValueError, match="no registrados"):
        build_game_features(
            warehouse, date(2024, 6, 1), date(2024, 6, 1), block_names=["nope"]
        )
