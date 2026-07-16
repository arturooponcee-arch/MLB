"""Tests del bloque Statcast del abridor (blocks.sp_statcast).

Valores a mano desde el fixture: pitcher 900 en g1 lanza 6 pitches —
whiffs 2, swings 4, called 1, FF (95+93+94)/3 = 94.0 mph,
xwOBA vs = (0.900 + 0.0) / 2 = 0.45. Su fila de g3 (ventana laggeada)
debe reproducir exactamente esos ratios.
"""

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.sp_statcast import SpStatcastBlock


@pytest.fixture
def result(warehouse: Database) -> pl.DataFrame:
    return SpStatcastBlock().build(warehouse)


def test_one_row_per_start(result: pl.DataFrame) -> None:
    # Todas las aperturas: 900 en g1 y g3, 950 en g1 (aunque g3 no tenga
    # pitch-by-pitch propio: las features salen de la historia).
    assert len(result) == 3
    assert result.select("game_pk", "team_id").n_unique() == 3


def test_first_start_is_null(result: pl.DataFrame) -> None:
    first = result.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
    assert first["sp_whiff_pct_3"][0] is None
    assert first["sp_fb_velo_3"][0] is None
    assert first["sp_xwoba_against_3"][0] is None


def test_lagged_window_hand_computed(result: pl.DataFrame) -> None:
    g3 = result.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 10))
    assert g3["sp_whiff_pct_3"][0] == pytest.approx(2 / 4)
    assert g3["sp_csw_pct_3"][0] == pytest.approx((1 + 2) / 6)
    assert g3["sp_fb_velo_3"][0] == pytest.approx(94.0)
    assert g3["sp_xwoba_against_3"][0] == pytest.approx(0.45)
    # Una sola apertura previa: las tres ventanas ven lo mismo.
    assert g3["sp_fb_velo_trend"][0] == pytest.approx(0.0)


def test_windows_share_data_with_single_prior_start(result: pl.DataFrame) -> None:
    g3 = result.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 10))
    assert g3["sp_whiff_pct_10"][0] == g3["sp_whiff_pct_3"][0]
    assert g3["sp_xwoba_against_5"][0] == g3["sp_xwoba_against_3"][0]


def test_suspended_game_keeps_real_starter(warehouse: Database) -> None:
    # Segundo "abridor" del mismo equipo en g1 con menos outs: debe perder.
    from datetime import datetime

    extra_line = pl.DataFrame(
        {
            "game_pk": [1],
            "team_id": [10],
            "player_id": [970],
            "is_starter": [True],
            "outs_recorded": [3],
            "pitches_thrown": [15],
            "batters_faced": [5],
            "strike_outs": [1],
            "base_on_balls": [0],
            "home_runs": [0],
            "earned_runs": [0],
        }
    )
    warehouse.upsert("pitching_lines", extra_line, keys=["game_pk", "player_id"])
    extra_pitch = pl.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [20],
            "pitch_number": [1],
            "game_date": [datetime(2024, 6, 1)],
            "pitcher": [970],
            "batter": [600],
            "p_throws": ["R"],
            "stand": ["R"],
            "inning_topbot": ["Top"],
            "pitch_type": ["FF"],
            "release_speed": [99.0],
            "description": ["ball"],
            "estimated_woba_using_speedangle": [None],
            "woba_value": [None],
            "woba_denom": [None],
        }
    )
    warehouse.upsert(
        "statcast_pitches", extra_pitch, keys=["game_pk", "at_bat_number", "pitch_number"]
    )
    result = SpStatcastBlock().build(warehouse)
    g1_team10 = result.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
    assert len(g1_team10) == 1  # dedup: gana 900 (18 outs > 3)
