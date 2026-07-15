"""Tests de cuotas manuales por CSV (betting.manual_odds)."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from mlb_quant.betting.manual_odds import load_manual_odds, write_template
from mlb_quant.betting.value import find_moneyline_value
from mlb_quant.ingestion.odds_api import ODDS_KEYS, ODDS_SCHEMA

NOW = datetime(2026, 7, 15, 18, 30, tzinfo=UTC)


@pytest.fixture
def games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1001, 1002],
            "game_datetime_utc": ["2026-07-15T23:05:00Z", "2026-07-16T01:45:00Z"],
            "home_team_name": ["Boston Red Sox", "San Francisco Giants"],
            "away_team_name": ["New York Yankees", "Los Angeles Dodgers"],
            "p_home_win": [0.52, 0.30],
        }
    )


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_builds_snapshot_schema(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home,book\n"
        "New York Yankees,Boston Red Sox,2.05,1.85,caliente\n"
        "Los Angeles Dodgers,San Francisco Giants,1.50,2.80,caliente\n",
    )
    odds = load_manual_odds(csv, games, now=NOW)
    assert odds.schema == pl.Schema(ODDS_SCHEMA)
    assert len(odds) == 4  # 2 juegos x 2 lados
    assert odds["snapshot_utc"].unique().to_list() == ["2026-07-15T18:30:00Z"]
    assert set(odds["event_id"].unique().to_list()) == {"manual-1001", "manual-1002"}
    assert odds["market"].unique().to_list() == ["h2h"]
    assert odds["book"].unique().to_list() == ["caliente"]
    sox = odds.filter(pl.col("outcome") == "Boston Red Sox")
    assert sox["decimal_odds"][0] == 1.85
    assert sox["commence_time_utc"][0] == "2026-07-15T23:05:00Z"
    # Claves de upsert únicas: el snapshot entra a odds_snapshots sin conflicto.
    assert odds.select(list(ODDS_KEYS)).n_unique() == len(odds)


def test_book_column_optional(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\nNew York Yankees,Boston Red Sox,2.05,1.85\n",
    )
    odds = load_manual_odds(csv, games, now=NOW)
    assert odds["book"].unique().to_list() == ["manual"]


def test_team_names_case_insensitive(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\n"
        "  new york yankees ,BOSTON RED SOX,2.05,1.85\n",
    )
    odds = load_manual_odds(csv, games, now=NOW)
    # Salen los nombres canónicos del calendario, no los tecleados.
    assert set(odds["outcome"].to_list()) == {"New York Yankees", "Boston Red Sox"}


def test_missing_column_raises(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away\nNew York Yankees,Boston Red Sox,2.05\n",
    )
    with pytest.raises(ValueError, match="odds_home"):
        load_manual_odds(csv, games, now=NOW)


def test_bad_odds_raise(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\nNew York Yankees,Boston Red Sox,0.95,1.85\n",
    )
    with pytest.raises(ValueError, match="Cuotas"):
        load_manual_odds(csv, games, now=NOW)


def test_empty_odds_raise(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\nNew York Yankees,Boston Red Sox,,1.85\n",
    )
    with pytest.raises(ValueError, match="Cuotas"):
        load_manual_odds(csv, games, now=NOW)


def test_duplicate_rows_raise(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\n"
        "New York Yankees,Boston Red Sox,2.05,1.85\n"
        "New York Yankees,Boston Red Sox,2.10,1.80\n",
    )
    with pytest.raises(ValueError, match="repetidas"):
        load_manual_odds(csv, games, now=NOW)


def test_unmatched_team_raises_with_names(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\nYankees,Boston Red Sox,2.05,1.85\n",
    )
    with pytest.raises(ValueError, match="nombres oficiales"):
        load_manual_odds(csv, games, now=NOW)


def test_doubleheader_row_applies_to_both_games(tmp_path: Path) -> None:
    games = pl.DataFrame(
        {
            "game_pk": [1001, 1003],
            "game_datetime_utc": ["2026-07-15T16:05:00Z", "2026-07-15T23:05:00Z"],
            "home_team_name": ["Boston Red Sox", "Boston Red Sox"],
            "away_team_name": ["New York Yankees", "New York Yankees"],
            "p_home_win": [0.52, 0.52],
        }
    )
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\nNew York Yankees,Boston Red Sox,2.05,1.85\n",
    )
    odds = load_manual_odds(csv, games, now=NOW)
    assert len(odds) == 4
    assert set(odds["event_id"].unique().to_list()) == {"manual-1001", "manual-1003"}


def test_loaded_odds_feed_value_detection(tmp_path: Path, games: pl.DataFrame) -> None:
    csv = _write_csv(
        tmp_path / "cuotas.csv",
        "away_team,home_team,odds_away,odds_home\n"
        "New York Yankees,Boston Red Sox,2.05,2.20\n"
        "Los Angeles Dodgers,San Francisco Giants,1.50,2.80\n",
    )
    odds = load_manual_odds(csv, games, now=NOW)
    value = find_moneyline_value(games, odds, min_ev=0.0)
    # Sox: 0.52 * 2.20 - 1 = +0.144; Dodgers: 0.70 * 1.50 - 1 = +0.05.
    assert value["team"].to_list() == ["Boston Red Sox", "Los Angeles Dodgers"]
    assert value["book"].unique().to_list() == ["manual"]


def test_write_template_lists_games(tmp_path: Path, games: pl.DataFrame) -> None:
    path = tmp_path / "plantilla.csv"
    n_games = write_template(path, games)
    assert n_games == 2
    template = pl.read_csv(path)
    assert template.columns == ["away_team", "home_team", "odds_away", "odds_home", "book"]
    assert template["home_team"].to_list() == ["Boston Red Sox", "San Francisco Giants"]
    assert template["odds_away"].null_count() == 2
    assert template["book"].to_list() == ["manual", "manual"]
