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


def test_elo_as_of_today(warehouse_with_scheduled: Database) -> None:
    from mlb_quant.feature_engineering.blocks.elo import ELO_GAMES_SQL, compute_elo

    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    _, ratings = compute_elo(warehouse_with_scheduled.query(ELO_GAMES_SQL))
    assert df["home_elo_pre"][0] == pytest.approx(ratings[10])
    assert df["away_elo_pre"][0] == pytest.approx(ratings[20])
    # El equipo 10 ganó todo: llega mejor rankeado.
    assert df["home_elo_pre"][0] > df["away_elo_pre"][0]


def test_sp_statcast_as_of_today(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # 900 solo tiene pitch-by-pitch en g1: whiff 2/4, velo FF 94.0.
    assert df["home_sp_whiff_pct_3"][0] == pytest.approx(0.5)
    assert df["home_sp_fb_velo_3"][0] == pytest.approx(94.0)
    assert df["home_sp_xwoba_against_3"][0] == pytest.approx(0.45)


def test_platoon_as_of_today(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # Visita (equipo 20) vs probable local 900 (R): 0.9/2 PA del 06-01.
    assert df["away_lineup_xwoba_vs_hand"][0] == pytest.approx(0.45)
    assert df["away_lineup_pa_vs_hand"][0] == 2
    # Local (equipo 10) vs probable visita 950 (L): 0.3/1 PA.
    assert df["home_lineup_xwoba_vs_hand"][0] == pytest.approx(0.30)
    assert df["home_lineup_pa_vs_hand"][0] == 1


def test_new_blocks_training_upcoming_parity(warehouse_with_scheduled: Database) -> None:
    """Guard contra drift: toda columna de los bloques nuevos que produce
    el entrenamiento debe existir también en las features de futuros."""
    from mlb_quant.feature_engineering.builder import build_game_features

    training = build_game_features(
        warehouse_with_scheduled,
        date(2023, 1, 1),
        date(2024, 12, 31),
        block_names=["elo", "rest_travel", "sp_statcast", "platoon"],
    )
    upcoming = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    base_columns = {
        "game_pk",
        "game_date",
        "season",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    }
    feature_columns = set(training.columns) - base_columns
    missing = sorted(feature_columns - set(upcoming.columns))
    assert not missing, f"Columnas de entrenamiento ausentes en upcoming: {missing}"


def test_rest_travel_as_of_today(warehouse_with_scheduled: Database) -> None:
    df = build_upcoming_features(
        warehouse_with_scheduled, date(2024, 6, 4), date(2024, 6, 4)
    )
    # Ambos jugaron g3 (2024-06-03, venue 100); el programado es en venue 100.
    assert df["home_team_days_rest"][0] == 1
    assert df["away_team_days_rest"][0] == 1
    assert df["home_travel_km"][0] == pytest.approx(0.0, abs=0.1)
    assert df["home_tz_shift"][0] == 0
    # Local: gira 0. Visitante: venía de jugar como visita (racha 1) -> 2.
    assert df["home_road_trip_len"][0] == 0
    assert df["away_road_trip_len"][0] == 2
