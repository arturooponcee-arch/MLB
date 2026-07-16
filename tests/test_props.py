"""Tests de props: agregación de bateador-juego y modelo Poisson."""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.models.props import PropModel, build_batter_frame, build_pitcher_k_frame
from mlb_quant.preprocessing.batter_games import build_batter_games


@pytest.fixture
def warehouse_with_statcast(warehouse: Database) -> Database:
    pitches = pl.DataFrame(
        {
            "game_pk": [1] * 6,
            "batter": [500, 500, 500, 500, 501, 501],
            "game_date": ["2024-06-01"] * 6,
            "at_bat_number": [1, 1, 20, 35, 2, 21],
            "pitch_number": [1, 2, 1, 1, 1, 1],
            "events": [None, "single", "home_run", "walk", "strikeout", "double"],
        }
    )
    warehouse.upsert(
        "statcast_pitches",
        pitches,
        keys=["game_pk", "batter", "at_bat_number", "pitch_number"],
    )
    return warehouse


class TestBatterGames:
    def test_aggregation(self, warehouse_with_statcast: Database) -> None:
        build_batter_games(warehouse_with_statcast)
        df = warehouse_with_statcast.query(
            "SELECT * FROM batter_games WHERE player_id = 500"
        )
        assert df["hits"][0] == 2  # single + home_run
        assert df["total_bases"][0] == 5  # 1 + 4
        assert df["walks"][0] == 1
        assert df["home_runs"][0] == 1
        assert df["pa"][0] == 3  # eventos no nulos

    def test_requires_statcast(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        # El warehouse base ya trae statcast_pitches: usar una DB vacía.
        empty = Database(tmp_path_factory.mktemp("sin_statcast") / "vacio.duckdb")
        with pytest.raises(ValueError, match="statcast_pitches"):
            build_batter_games(empty)


class TestPropModel:
    @staticmethod
    def _frame(n: int = 400) -> pl.DataFrame:
        rng = np.random.default_rng(5)
        skill = rng.normal(0, 1, n)
        return pl.DataFrame(
            {
                "game_pk": np.arange(n),
                "game_date": [date(2024, 4, 1) + timedelta(days=i // 5) for i in range(n)],
                "season": [2024] * n,
                "player_id": [1] * n,
                "team_id": [1] * n,
                "skill": skill,
                "target": rng.poisson(np.exp(1.6 + 0.3 * skill)),
            }
        )

    def test_learns_signal(self) -> None:
        frame = self._frame()
        model = PropModel(alpha=0.1).fit(frame)
        lam = model.predict_lambda(frame)["lambda"].to_numpy()
        assert np.corrcoef(lam, frame["skill"].to_numpy())[0, 1] > 0.8

    def test_probability_over_analytic(self) -> None:
        # P(X > 5.5) con lambda=5: 1 - CDF(5).
        lam = 5.0
        cdf5 = sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(6))
        result = PropModel.probability_over(np.array([lam]), 5.5)
        assert result[0] == pytest.approx(1 - cdf5, abs=1e-6)

    def test_missing_target_raises(self) -> None:
        with pytest.raises(ValueError, match="target"):
            PropModel().fit(self._frame().drop("target"))


class TestFrames:
    def test_pitcher_k_frame_from_features(self, warehouse: Database) -> None:
        """features_game sintética mínima + pitching_lines del conftest."""
        features = pl.DataFrame(
            {
                "game_pk": [1],
                "game_date": [date(2024, 6, 1)],
                "season": [2024],
                "home_team_id": [10],
                "away_team_id": [20],
                "home_sp_k_pct_5": [0.25],
                "away_sp_k_pct_5": [0.18],
                "home_spq_whiff_pct": [30.0],
                "away_spq_whiff_pct": [None],
                "home_lineup_whiff_pct": [21.0],
                "away_lineup_whiff_pct": [45.0],
                "park_elevation": [10.0],
            }
        )
        warehouse.upsert("features_game", features, keys=["game_pk"])
        frame = build_pitcher_k_frame(warehouse)
        assert frame.height == 2  # ambos abridores del juego 1
        home = frame.filter(pl.col("team_id") == 10)
        assert home["target"][0] == 6  # K de Gerrit Cole en conftest
        assert home["opp_lineup_whiff_pct"][0] == 45.0  # lineup RIVAL
        assert home["sp_k_pct_5"][0] == 0.25

    def test_batter_frame(self, warehouse_with_statcast: Database) -> None:
        build_batter_games(warehouse_with_statcast)
        frame = build_batter_frame(warehouse_with_statcast, "hits")
        row = frame.filter(pl.col("player_id") == 500)
        assert row["target"][0] == 2
        assert row["bat_xwoba"][0] == pytest.approx(0.400)  # Savant 2023
        # Abridor rival del equipo 10 es el 950 (equipo 20): sin Savant.
        assert row["opp_sp_xwoba"][0] is None
        assert row["batting_order"][0] == 1

    def test_batter_frame_invalid_stat(self, warehouse: Database) -> None:
        with pytest.raises(ValueError, match="no soportada"):
            build_batter_frame(warehouse, "robos")
