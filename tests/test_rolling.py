"""Tests de utilidades rolling anti-leakage."""

from datetime import date

import polars as pl

from mlb_quant.feature_engineering.rolling import (
    RollingSpec,
    add_days_since_previous,
    add_lagged_rolling,
)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["A", "A", "A", "B"],
            "day": [1, 2, 3, 1],
            "runs": [4, 6, 2, 9],
        }
    )


def test_first_row_of_group_is_null() -> None:
    result = add_lagged_rolling(
        _events(),
        group_by="team",
        sort_by="day",
        specs=[RollingSpec("runs", 2, "mean", "runs_avg_2")],
    )
    team_a = result.filter(pl.col("team") == "A").sort("day")
    assert team_a["runs_avg_2"][0] is None  # sin historia previa


def test_rolling_mean_excludes_current_row() -> None:
    result = add_lagged_rolling(
        _events(),
        group_by="team",
        sort_by="day",
        specs=[RollingSpec("runs", 2, "mean", "runs_avg_2")],
    )
    team_a = result.filter(pl.col("team") == "A").sort("day")
    assert team_a["runs_avg_2"][1] == 4.0  # solo día 1
    assert team_a["runs_avg_2"][2] == 5.0  # días 1-2: (4+6)/2


def test_rolling_sum_and_group_isolation() -> None:
    result = add_lagged_rolling(
        _events(),
        group_by="team",
        sort_by="day",
        specs=[RollingSpec("runs", 3, "sum", "runs_sum_3")],
    )
    team_b = result.filter(pl.col("team") == "B")
    assert team_b["runs_sum_3"][0] is None  # el historial de A no contamina a B


def test_days_since_previous() -> None:
    df = pl.DataFrame(
        {
            "player": ["X", "X", "X"],
            "game_date": [date(2024, 6, 1), date(2024, 6, 6), date(2024, 6, 8)],
        }
    )
    result = add_days_since_previous(
        df, group_by="player", date_column="game_date", alias="rest"
    )
    assert result["rest"].to_list() == [None, 5, 2]
