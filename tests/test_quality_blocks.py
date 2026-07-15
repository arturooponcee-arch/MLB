"""Tests de los bloques de calidad (lineup, abridor, bullpen)."""

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.bullpen import BullpenFatigueBlock
from mlb_quant.feature_engineering.blocks.lineup_quality import LineupQualityBlock
from mlb_quant.feature_engineering.blocks.sp_quality import SpQualityBlock


class TestLineupQuality:
    def test_averages_prior_season_of_starters(self, warehouse: Database) -> None:
        df = LineupQualityBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
        # Titulares 500 y 501 con temporada 2023 (juego de 2024).
        assert row["lineup_xwoba"][0] == pytest.approx((0.400 + 0.360) / 2)
        assert row["lineup_k_pct"][0] == pytest.approx((20.0 + 15.0) / 2)
        assert row["lineup_known_batters"][0] == 2

    def test_low_pa_batters_excluded(self, warehouse: Database) -> None:
        df = LineupQualityBlock().build(warehouse)
        # Titular 600 tiene solo 50 PA en 2023 -> fuera del promedio.
        row = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 20))
        assert row["lineup_known_batters"][0] == 0
        assert row["lineup_xwoba"][0] is None

    def test_substitutes_ignored(self, warehouse: Database) -> None:
        df = LineupQualityBlock().build(warehouse)
        # Jugador 601 no es titular: el equipo 20 solo cuenta al 600.
        assert df.filter(pl.col("team_id") == 20).height == 1


class TestSpQuality:
    def test_starter_prior_season_stats(self, warehouse: Database) -> None:
        df = SpQualityBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
        assert row["spq_xwoba_against"][0] == pytest.approx(0.290)
        assert row["spq_xera"][0] == pytest.approx(3.50)

    def test_rookie_starter_is_null(self, warehouse: Database) -> None:
        df = SpQualityBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 20))
        assert row["spq_xwoba_against"][0] is None

    def test_unique_per_team_game(self, warehouse: Database) -> None:
        df = SpQualityBlock().build(warehouse)
        assert df.select("game_pk", "team_id").is_duplicated().sum() == 0


class TestBullpenFatigue:
    def test_pitches_previous_days(self, warehouse: Database) -> None:
        df = BullpenFatigueBlock().build(warehouse)
        # Juego 3 (2024-06-03), equipo 10: relevo del 1 jun (25) y 2 jun (30).
        row = df.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 10))
        assert row["bullpen_pitches_1d"][0] == 30
        assert row["bullpen_pitches_3d"][0] == 55
        assert row["bullpen_relievers_1d"][0] == 1

    def test_no_prior_relief_is_zero(self, warehouse: Database) -> None:
        df = BullpenFatigueBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
        assert row["bullpen_pitches_3d"][0] == 0

    def test_only_looks_backwards(self, warehouse: Database) -> None:
        df = BullpenFatigueBlock().build(warehouse)
        # Juego 2 (2024-06-02), equipo 10: solo cuenta el relevo del 1 jun.
        row = df.filter((pl.col("game_pk") == 2) & (pl.col("team_id") == 10))
        assert row["bullpen_pitches_1d"][0] == 25
        assert row["bullpen_pitches_3d"][0] == 25
