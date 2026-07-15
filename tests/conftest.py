"""Fixtures compartidas: warehouse sintético para tests de features."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from mlb_quant.db import Database


@pytest.fixture
def warehouse(tmp_path: Path) -> Database:
    """Warehouse mínimo pero completo: 2 temporadas, 2 estadios, clima y pitcheo.

    Diseñado para verificar a mano cada feature:
    - Equipo 10: gana g1 (5-3, local) y g2 (7-2, visita); juega g3 local.
    - Pitcher 900: abre g1 y g3 (2 días de descanso), línea conocida en g1.
    - Venue 100: azimuth 0 (center field al norte), abierto; venue 200: domo.
    - 2023: venue 100 promedia 10 carreras/juego, venue 200 promedia 6.
    - Clima en g1: viento 10 km/h desde el sur (180°) -> sopla hacia el norte.
    """
    db = Database(tmp_path / "warehouse.duckdb")

    games = pl.DataFrame(
        {
            "game_pk": [901, 902, 1, 2, 3],
            "game_date": [
                date(2023, 6, 1),
                date(2023, 6, 2),
                date(2024, 6, 1),
                date(2024, 6, 2),
                date(2024, 6, 3),
            ],
            "game_datetime_utc": [
                "2023-06-01T23:05:00Z",
                "2023-06-02T23:05:00Z",
                "2024-06-01T23:05:00Z",
                "2024-06-02T23:05:00Z",
                "2024-06-03T23:05:00Z",
            ],
            "season": [2023, 2023, 2024, 2024, 2024],
            "game_type": ["R", "R", "R", "R", "R"],
            "status": ["Final", "Final", "Final", "Final", "Final"],
            "venue_id": [100, 200, 100, 200, 100],
            "home_team_id": [10, 20, 10, 20, 10],
            "away_team_id": [20, 10, 20, 10, 20],
            "home_score": [6, 2, 5, 2, 6],
            "away_score": [4, 4, 3, 7, 2],
        }
    )
    db.upsert("games", games, keys=["game_pk"])

    venues = pl.DataFrame(
        {
            "venue_id": [100, 200, 100, 200],
            "season": [2023, 2023, 2024, 2024],
            "azimuth_angle": [0.0, 90.0, 0.0, 90.0],
            "roof_type": ["Open", "Dome", "Open", "Dome"],
            "elevation": [10.0, 500.0, 10.0, 500.0],
            "center": [400, 410, 400, 410],
            "left_line": [330, 335, 330, 335],
            "right_line": [325, 330, 325, 330],
        }
    )
    db.upsert("venues", venues, keys=["venue_id", "season"])

    weather = pl.DataFrame(
        {
            "venue_id": [100],
            "time_utc": ["2024-06-01T23:00"],
            "temperature_c": [25.0],
            "humidity_pct": [60.0],
            "wind_speed_kmh": [10.0],
            "wind_direction_deg": [180.0],
            "pressure_hpa": [1013.0],
        }
    )
    db.upsert("weather_hourly", weather, keys=["venue_id", "time_utc"])

    pitching = pl.DataFrame(
        {
            "game_pk": [1, 1, 3, 1, 2],
            "team_id": [10, 20, 10, 10, 10],
            "player_id": [900, 950, 900, 960, 961],
            "is_starter": [True, True, True, False, False],
            "outs_recorded": [18, 15, 12, 9, 6],
            "pitches_thrown": [90, 80, 70, 25, 30],
            "batters_faced": [24, 22, 18, 10, 8],
            "strike_outs": [6, 4, 3, 2, 1],
            "base_on_balls": [2, 3, 2, 1, 1],
            "home_runs": [1, 0, 1, 0, 0],
            "earned_runs": [3, 2, 4, 1, 0],
        }
    )
    db.upsert("pitching_lines", pitching, keys=["game_pk", "player_id"])

    lineups = pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 1],
            "team_id": [10, 10, 20, 20],
            "player_id": [500, 501, 600, 601],
            "batting_order": [1, 2, 1, 2],
            "is_starting_lineup": [True, True, True, False],
        }
    )
    db.upsert("lineups", lineups, keys=["game_pk", "player_id"])

    savant_metrics = pl.DataFrame(
        {
            "player_id": [500, 501, 600, 900],
            "year": [2023, 2023, 2023, 2023],
            "player_type": ["batter", "batter", "batter", "pitcher"],
            "pa": [600, 550, 50, 700],  # 600 no llega al mínimo de 100 PA
            "xwoba": [0.400, 0.360, 0.500, 0.290],
            "woba": [0.390, 0.350, 0.480, 0.300],
            "k_percent": [20.0, 15.0, 40.0, 25.0],
            "bb_percent": [12.0, 8.0, 2.0, 7.0],
            "barrel_batted_rate": [15.0, 9.0, 1.0, 6.0],
            "hard_hit_percent": [50.0, 42.0, 20.0, 35.0],
            "whiff_percent": [24.0, 18.0, 45.0, 30.0],
            "oz_swing_percent": [26.0, 30.0, 45.0, 28.0],
        }
    )
    db.upsert(
        "savant_statcast_metrics", savant_metrics, keys=["player_id", "year", "player_type"]
    )

    savant_expected = pl.DataFrame(
        {
            "player_id": [900],
            "year": [2023],
            "player_type": ["pitcher"],
            "pa": [700],
            "xera": [3.50],
        }
    )
    db.upsert(
        "savant_expected_stats", savant_expected, keys=["player_id", "year", "player_type"]
    )

    return db
