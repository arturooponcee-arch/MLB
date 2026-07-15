"""Tests de los bloques de features sobre el warehouse sintético."""

import polars as pl
import pytest

import mlb_quant.feature_engineering.blocks  # noqa: F401  (registra los bloques)
from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.park import ParkBlock
from mlb_quant.feature_engineering.blocks.pitcher_form import PitcherFormBlock
from mlb_quant.feature_engineering.blocks.team_form import TeamFormBlock
from mlb_quant.feature_engineering.blocks.weather import WeatherBlock


class TestTeamForm:
    def test_no_history_gives_null(self, warehouse: Database) -> None:
        df = TeamFormBlock().build(warehouse)
        first = df.filter((pl.col("game_pk") == 901) & (pl.col("team_id") == 10))
        assert first["team_win_rate_5"][0] is None

    def test_win_rate_and_runs_before_game3(self, warehouse: Database) -> None:
        df = TeamFormBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 10))
        # Historia 2023+2024 del equipo 10: g901 W(6-4), g902 W(4-2), g1 W(5-3), g2 W(7-2)
        assert row["team_win_rate_5"][0] == 1.0
        assert row["team_runs_scored_5"][0] == pytest.approx((6 + 4 + 5 + 7) / 4)
        assert row["team_runs_allowed_5"][0] == pytest.approx((4 + 2 + 3 + 2) / 4)


class TestPitcherForm:
    def test_first_start_has_null_rolling(self, warehouse: Database) -> None:
        df = PitcherFormBlock().build(warehouse)
        first = df.filter((pl.col("game_pk") == 1) & (pl.col("team_id") == 10))
        assert first["sp_fip_3"][0] is None
        assert first["sp_days_rest"][0] is None

    def test_second_start_features(self, warehouse: Database) -> None:
        df = PitcherFormBlock().build(warehouse)
        row = df.filter((pl.col("game_pk") == 3) & (pl.col("team_id") == 10))
        # Historia = solo la apertura del juego 1: 18 outs, 24 BF, 6 K, 2 BB, 1 HR, 3 ER.
        assert row["sp_days_rest"][0] == 2
        assert row["sp_k_pct_3"][0] == pytest.approx(6 / 24)
        assert row["sp_bb_pct_3"][0] == pytest.approx(2 / 24)
        assert row["sp_pitches_avg_3"][0] == pytest.approx(90.0)
        assert row["sp_ip_per_start_3"][0] == pytest.approx(6.0)
        assert row["sp_fip_3"][0] == pytest.approx((13 * 1 + 3 * 2 - 2 * 6) / 6 + 3.15)
        assert row["sp_era_3"][0] == pytest.approx(9 * 3 / 6)

    def test_one_row_per_team_start(self, warehouse: Database) -> None:
        df = PitcherFormBlock().build(warehouse)
        assert df.filter(pl.col("game_pk") == 1).height == 2  # ambos abridores


class TestPark:
    def test_physical_features(self, warehouse: Database) -> None:
        df = ParkBlock().build(warehouse)
        game1 = df.filter(pl.col("game_pk") == 1)
        assert game1["park_elevation"][0] == 10.0
        assert game1["park_has_roof"][0] == 0
        game2 = df.filter(pl.col("game_pk") == 2)
        assert game2["park_has_roof"][0] == 1

    def test_run_factor_uses_only_past_seasons(self, warehouse: Database) -> None:
        df = ParkBlock().build(warehouse)
        # 2023 no tiene temporadas previas -> null.
        assert df.filter(pl.col("game_pk") == 901)["park_run_factor"][0] is None
        # 2024 en venue 100: 2023 dio 10 r/j allí; liga 2023 = (10+6)/2 = 8.
        game1 = df.filter(pl.col("game_pk") == 1)
        assert game1["park_run_factor"][0] == pytest.approx(10 / 8)


class TestWeather:
    def test_wind_out_component(self, warehouse: Database) -> None:
        df = WeatherBlock().build(warehouse)
        game1 = df.filter(pl.col("game_pk") == 1)
        # Viento DESDE 180° (sur) -> sopla hacia 0° (norte) = azimuth -> todo out.
        assert game1["wx_wind_out"][0] == pytest.approx(10.0, abs=1e-6)
        assert game1["wx_wind_cross"][0] == pytest.approx(0.0, abs=1e-6)
        assert game1["wx_temperature_c"][0] == 25.0

    def test_game_without_weather_is_null(self, warehouse: Database) -> None:
        df = WeatherBlock().build(warehouse)
        game2 = df.filter(pl.col("game_pk") == 2)
        assert game2["wx_temperature_c"][0] is None
        assert game2["wx_roof_possible"][0] == 1
