"""Tests de detección de valor moneyline (betting.value)."""

import polars as pl
import pytest

from mlb_quant.betting.value import (
    best_moneyline_prices,
    find_moneyline_value,
    latest_snapshot,
)


def _odds_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "snapshot_utc": "2026-07-15T18:30:00Z",
        "event_id": "evt-1",
        "commence_time_utc": "2026-07-15T23:05:00Z",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "book": "draftkings",
        "market": "h2h",
        "outcome": "Boston Red Sox",
        "point": None,
        "decimal_odds": 2.1,
        "last_update_utc": "2026-07-15T18:00:00Z",
    }
    base.update(overrides)
    return base


@pytest.fixture
def odds() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _odds_row(),
            _odds_row(outcome="New York Yankees", decimal_odds=1.78),
            _odds_row(book="fanduel", decimal_odds=2.2),
            _odds_row(book="fanduel", outcome="New York Yankees", decimal_odds=1.72),
            _odds_row(book="draftkings", market="totals", outcome="Over", point=8.5),
        ]
    )


@pytest.fixture
def games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1001],
            "game_datetime_utc": ["2026-07-15T23:05:00Z"],
            "home_team_name": ["Boston Red Sox"],
            "away_team_name": ["New York Yankees"],
            "p_home_win": [0.52],
        }
    )


def test_latest_snapshot_keeps_newest(odds: pl.DataFrame) -> None:
    stale = pl.DataFrame([_odds_row(snapshot_utc="2026-07-15T10:00:00Z", decimal_odds=3.0)])
    combined = pl.concat([odds, stale])
    result = latest_snapshot(combined)
    assert result["snapshot_utc"].unique().to_list() == ["2026-07-15T18:30:00Z"]


def test_best_prices_takes_max_across_books(odds: pl.DataFrame) -> None:
    best = best_moneyline_prices(odds)
    assert len(best) == 2
    sox = best.filter(pl.col("outcome") == "Boston Red Sox")
    assert sox["decimal_odds"][0] == 2.2
    assert sox["book"][0] == "fanduel"
    yankees = best.filter(pl.col("outcome") == "New York Yankees")
    assert yankees["decimal_odds"][0] == 1.78
    assert yankees["book"][0] == "draftkings"


def test_best_prices_ignores_totals(odds: pl.DataFrame) -> None:
    best = best_moneyline_prices(odds)
    assert "Over" not in best["outcome"].to_list()


def test_find_value_reports_positive_ev_side(games: pl.DataFrame, odds: pl.DataFrame) -> None:
    value = find_moneyline_value(games, odds, bankroll=1000.0, min_ev=0.0)
    # Home: 0.52 * 2.2 - 1 = +0.144. Away: 0.48 * 1.78 - 1 = -0.1456.
    assert len(value) == 1
    row = value.row(0, named=True)
    assert row["side"] == "home"
    assert row["team"] == "Boston Red Sox"
    assert row["best_odds"] == 2.2
    assert row["book"] == "fanduel"
    assert row["ev"] == pytest.approx(0.144)
    assert row["fair_odds"] == pytest.approx(1 / 0.52)
    assert row["kelly_stake"] > 0


def test_find_value_respects_min_ev(games: pl.DataFrame, odds: pl.DataFrame) -> None:
    value = find_moneyline_value(games, odds, min_ev=0.20)
    assert value.is_empty()


def test_find_value_respects_bankroll_cap(games: pl.DataFrame, odds: pl.DataFrame) -> None:
    value = find_moneyline_value(games, odds, bankroll=1000.0, max_fraction=0.01, min_ev=0.0)
    row = value.row(0, named=True)
    assert row["capped"] is True
    assert row["kelly_stake"] == pytest.approx(10.0)


def test_unmatched_game_is_dropped(odds: pl.DataFrame) -> None:
    games = pl.DataFrame(
        {
            "game_pk": [2002],
            "game_datetime_utc": ["2026-07-15T23:05:00Z"],
            "home_team_name": ["Chicago Cubs"],
            "away_team_name": ["St. Louis Cardinals"],
            "p_home_win": [0.60],
        }
    )
    assert find_moneyline_value(games, odds).is_empty()


def test_event_outside_time_tolerance_is_dropped(odds: pl.DataFrame) -> None:
    games = pl.DataFrame(
        {
            "game_pk": [1001],
            "game_datetime_utc": ["2026-07-17T23:05:00Z"],
            "home_team_name": ["Boston Red Sox"],
            "away_team_name": ["New York Yankees"],
            "p_home_win": [0.52],
        }
    )
    assert find_moneyline_value(games, odds).is_empty()


def test_doubleheader_matches_by_time_proximity(odds: pl.DataFrame) -> None:
    game2 = pl.DataFrame(
        [
            _odds_row(event_id="evt-2", commence_time_utc="2026-07-15T16:05:00Z"),
            _odds_row(
                event_id="evt-2",
                commence_time_utc="2026-07-15T16:05:00Z",
                outcome="New York Yankees",
                decimal_odds=1.9,
            ),
        ]
    )
    combined = pl.concat([odds, game2])
    games = pl.DataFrame(
        {
            "game_pk": [1001, 1002],
            "game_datetime_utc": ["2026-07-15T23:05:00Z", "2026-07-15T16:05:00Z"],
            "home_team_name": ["Boston Red Sox", "Boston Red Sox"],
            "away_team_name": ["New York Yankees", "New York Yankees"],
            "p_home_win": [0.52, 0.52],
        }
    )
    value = find_moneyline_value(games, combined, min_ev=-1.0)
    # Cada juego empareja con su propio evento: 2 juegos x 2 lados.
    assert len(value) == 4
    assert value["game_pk"].n_unique() == 2


def test_empty_inputs_give_empty_frame(games: pl.DataFrame, odds: pl.DataFrame) -> None:
    assert find_moneyline_value(pl.DataFrame(), odds).is_empty()
    assert find_moneyline_value(games, pl.DataFrame()).is_empty()
