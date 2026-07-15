"""Tests de las features de juegos programados (as-of)."""

from datetime import date

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.upcoming import build_upcoming_features


@pytest.fixture
def warehouse_with_scheduled(warehouse: Database) -> Database:
    """Añade un juego programado (2024-06-04) con probables al warehouse."""
    scheduled = pl.DataFrame(
        {
            "game_pk": [99],
            "game_date": [date(2024, 6, 4)],
            "game_datetime_utc": ["2024-06-04T23:05:00Z"],
            "season": [2024],
            "game_type": ["R"],
            "status": ["Scheduled"],
            "venue_id": [100],
            "home_team_id": [10],
            "away_team_id": [20],
            "home_score": [None],
            "away_score": [None],
            "venue_name": ["Estadio Cien"],
            "home_team_name": ["Locales"],
            "away_team_name": ["Visitantes"],
            "home_probable_pitcher_id": [900],
            "away_probable_pitcher_id": [950],
            "home_probable_pitcher_name": ["Abridor Local"],
            "away_probable_pitcher_name": ["Abridor Visita"],
        }
    )
    warehouse.upsert("games", scheduled, keys=["game_pk"])
    return warehouse


def test_one_row_per_scheduled_game(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    assert df.height == 1
    assert df["game_pk"][0] == 99


def test_team_form_as_of_today(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # Equipo 10 ganó sus 5 juegos completados (2 de 2023 + 3 de 2024).
    assert df["home_team_win_rate_5"][0] == 1.0
    # Equipo 20 los perdió todos.
    assert df["away_team_win_rate_5"][0] == 0.0


def test_pitcher_rest_and_quality(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # 900 lanzó por última vez el 3 de junio -> 1 día de descanso.
    assert df["home_sp_days_rest"][0] == 1
    # Calidad de temporada previa (2023) del probable local.
    assert df["home_spq_xwoba_against"][0] == pytest.approx(0.290)
    # El probable visitante (950) no tiene temporada 2023 en Savant.
    assert df["away_spq_xwoba_against"][0] is None


def test_pitcher_form_includes_last_start(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # As-of: las DOS aperturas de 900 cuentan (g1: 6K/24BF, g3: 3K/18BF).
    assert df["home_sp_k_pct_3"][0] == pytest.approx((6 + 3) / (24 + 18))


def test_game_grain_blocks_join(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    assert "park_elevation" in df.columns
    assert df["park_elevation"][0] == 10.0
    assert "home_bullpen_pitches_3d" in df.columns


def test_empty_range_returns_empty(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 7, 1), date(2024, 7, 1)
    )
    assert df.is_empty()
