"""Tests del bloque de descanso y viaje (blocks.rest_travel).

Con el fixture ``warehouse``: equipo 10 juega 901 (v100, local),
902 (v200, visita), g1 (v100, local), g2 (v200, visita), g3 (v100, local).
Venue 100 = (40, -74), venue 200 = (34, -118):
haversine = 3923.7 km, husos -5 y -8.
"""

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.rest_travel import RestTravelBlock

#: Distancia (40,-74) -> (34,-118) calculada a mano con la fórmula haversine.
_EXPECTED_KM = 3923.7


@pytest.fixture
def result(warehouse: Database) -> pl.DataFrame:
    return RestTravelBlock().build(warehouse)


def _row(result: pl.DataFrame, game_pk: int, team_id: int) -> dict:
    return result.filter(
        (pl.col("game_pk") == game_pk) & (pl.col("team_id") == team_id)
    ).row(0, named=True)


def test_one_row_per_team_game(result: pl.DataFrame) -> None:
    assert len(result) == 10
    assert result.select("game_pk", "team_id").n_unique() == 10
    assert set(result.columns) == {
        "game_pk",
        "team_id",
        "team_days_rest",
        "road_trip_len",
        "travel_km",
        "tz_shift",
    }


def test_first_game_has_null_rest_and_travel(result: pl.DataFrame) -> None:
    first = _row(result, 901, 10)
    assert first["team_days_rest"] is None
    assert first["travel_km"] is None
    assert first["tz_shift"] is None


def test_days_rest_hand_computed(result: pl.DataFrame) -> None:
    assert _row(result, 902, 10)["team_days_rest"] == 1  # 06-02 - 06-01
    assert _row(result, 1, 10)["team_days_rest"] == 365  # 2024-06-01 - 2023-06-02
    assert _row(result, 3, 10)["team_days_rest"] == 1  # 06-03 - 06-02


def test_travel_km_haversine(result: pl.DataFrame) -> None:
    # 902: el equipo 10 viaja de venue 100 a venue 200.
    assert _row(result, 902, 10)["travel_km"] == pytest.approx(_EXPECTED_KM, abs=1.0)
    # g3: regresa de venue 200 a venue 100 (misma distancia).
    assert _row(result, 3, 10)["travel_km"] == pytest.approx(_EXPECTED_KM, abs=1.0)


def test_tz_shift_signed(result: pl.DataFrame) -> None:
    # Este (-5) -> oeste (-8): -3 horas; el regreso: +3.
    assert _row(result, 902, 10)["tz_shift"] == -3
    assert _row(result, 3, 10)["tz_shift"] == 3


def test_road_trip_len(result: pl.DataFrame) -> None:
    # Equipo 10: local, visita, local, visita, local -> 0,1,0,1,0.
    assert _row(result, 901, 10)["road_trip_len"] == 0
    assert _row(result, 902, 10)["road_trip_len"] == 1
    assert _row(result, 3, 10)["road_trip_len"] == 0
    # Equipo 20: visita, local, visita, local, visita -> gira nunca acumula.
    assert _row(result, 901, 20)["road_trip_len"] == 1
    assert _row(result, 3, 20)["road_trip_len"] == 1


def test_road_trip_accumulates_consecutive_away(warehouse: Database) -> None:
    from datetime import date

    extra = pl.DataFrame(
        {
            "game_pk": [4, 5],
            "game_date": [date(2024, 6, 4), date(2024, 6, 5)],
            "game_datetime_utc": ["2024-06-04T23:05:00Z", "2024-06-05T23:05:00Z"],
            "season": [2024, 2024],
            "game_type": ["R", "R"],
            "status": ["Final", "Final"],
            "venue_id": [200, 200],
            "home_team_id": [20, 20],
            "away_team_id": [10, 10],
            "home_score": [1, 2],
            "away_score": [3, 4],
        }
    )
    warehouse.upsert("games", extra, keys=["game_pk"])
    result = RestTravelBlock().build(warehouse)
    # Equipo 10 visita dos juegos seguidos: la gira acumula 1, 2.
    assert _row(result, 4, 10)["road_trip_len"] == 1
    assert _row(result, 5, 10)["road_trip_len"] == 2
