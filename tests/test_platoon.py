"""Tests del bloque platoon (blocks.platoon).

Del fixture: en g1 el equipo 20 batea contra 900 (R) y produce
0.9 wOBA-num en 2 PA; el equipo 10 batea contra 950 (L): 0.3 en 1 PA.
En g3 el abridor rival del equipo 20 vuelve a ser 900 (R): su ventana
de 365 días previa contiene exactamente esas 2 PA vs R.
"""

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.platoon import PlatoonBlock


@pytest.fixture
def result(warehouse: Database) -> pl.DataFrame:
    return PlatoonBlock().build(warehouse)


def _row(result: pl.DataFrame, game_pk: int, team_id: int) -> dict:
    return result.filter(
        (pl.col("game_pk") == game_pk) & (pl.col("team_id") == team_id)
    ).row(0, named=True)


def test_rows_only_where_opposing_hand_known(result: pl.DataFrame) -> None:
    # g1: ambos equipos (900 y 950 con mano conocida). g3: solo el equipo
    # 20 (el rival del 10 no tiene abridor en pitching_lines).
    keys = set(result.select("game_pk", "team_id").iter_rows())
    assert keys == {(1, 10), (1, 20), (3, 20)}


def test_no_history_gives_null_and_zero_pa(result: pl.DataFrame) -> None:
    g1_team20 = _row(result, 1, 20)
    assert g1_team20["lineup_xwoba_vs_hand"] is None
    assert g1_team20["lineup_pa_vs_hand"] == 0


def test_trailing_window_hand_computed(result: pl.DataFrame) -> None:
    # Equipo 20 vs R en g3: sus 2 PA vs 900 (R) del 06-01 -> 0.9/2.
    g3_team20 = _row(result, 3, 20)
    assert g3_team20["lineup_xwoba_vs_hand"] == pytest.approx(0.45)
    assert g3_team20["lineup_pa_vs_hand"] == 2


def test_same_day_pas_are_excluded(result: pl.DataFrame) -> None:
    # En g1 las PA del propio día NO cuentan (ventana estricta < fecha).
    assert _row(result, 1, 10)["lineup_xwoba_vs_hand"] is None
    assert _row(result, 1, 20)["lineup_xwoba_vs_hand"] is None


def test_split_filters_by_hand(warehouse: Database) -> None:
    # Un abridor zurdo nuevo en g3 contra el equipo 20: su ventana vs L
    # está vacía (las 2 PA previas del equipo 20 fueron vs R).
    lefty_line = pl.DataFrame(
        {
            "game_pk": [3],
            "team_id": [10],
            "player_id": [950],
            "is_starter": [True],
            "outs_recorded": [21],
            "pitches_thrown": [95],
            "batters_faced": [26],
            "strike_outs": [5],
            "base_on_balls": [2],
            "home_runs": [0],
            "earned_runs": [2],
        }
    )
    warehouse.upsert("pitching_lines", lefty_line, keys=["game_pk", "player_id"])
    result = PlatoonBlock().build(warehouse)
    g3_team20 = result.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 20))
    # 950 (con 21 outs) desplaza a 900 (12) como abridor del equipo 10:
    # el rival del equipo 20 pasa a ser zurdo -> sin PAs previas vs L.
    assert g3_team20["lineup_xwoba_vs_hand"][0] is None
    assert g3_team20["lineup_pa_vs_hand"][0] == 0
