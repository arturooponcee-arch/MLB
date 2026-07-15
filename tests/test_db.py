"""Tests de la capa DuckDB."""

from pathlib import Path

import polars as pl
import pytest

from mlb_quant.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.duckdb")


def _sample(scores: tuple[int, int] = (5, 3)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1, 2],
            "home_score": list(scores),
            "status": ["Final", "Final"],
        }
    )


def test_upsert_creates_table(db: Database) -> None:
    written = db.upsert("games", _sample(), keys=["game_pk"])
    assert written == 2
    assert db.table_exists("games")
    assert db.query("SELECT COUNT(*) AS n FROM games")["n"][0] == 2


def test_upsert_is_idempotent(db: Database) -> None:
    db.upsert("games", _sample(), keys=["game_pk"])
    db.upsert("games", _sample(), keys=["game_pk"])
    assert db.query("SELECT COUNT(*) AS n FROM games")["n"][0] == 2


def test_upsert_replaces_matching_rows(db: Database) -> None:
    db.upsert("games", _sample(scores=(5, 3)), keys=["game_pk"])
    db.upsert("games", _sample(scores=(7, 3)), keys=["game_pk"])
    result = db.query("SELECT home_score FROM games WHERE game_pk = 1")
    assert result["home_score"][0] == 7
    assert db.query("SELECT COUNT(*) AS n FROM games")["n"][0] == 2


def test_upsert_empty_frame_is_noop(db: Database) -> None:
    written = db.upsert("games", _sample().clear(), keys=["game_pk"])
    assert written == 0
    assert not db.table_exists("games")


def test_upsert_missing_key_raises(db: Database) -> None:
    with pytest.raises(ValueError, match="Claves ausentes"):
        db.upsert("games", _sample(), keys=["no_existe"])


def test_upsert_duplicate_incoming_keys_raises(db: Database) -> None:
    duplicated = pl.concat([_sample(), _sample()])
    with pytest.raises(ValueError, match="duplicadas"):
        db.upsert("games", duplicated, keys=["game_pk"])


def test_upsert_evolves_schema_with_new_columns(db: Database) -> None:
    db.upsert("games", _sample(), keys=["game_pk"])
    wider = pl.DataFrame(
        {"game_pk": [3], "home_score": [9], "status": ["Final"], "xera": [3.21]}
    )
    db.upsert("games", wider, keys=["game_pk"])
    result = db.query("SELECT game_pk, xera FROM games ORDER BY game_pk")
    assert result["xera"].to_list() == [None, None, 3.21]


def test_upsert_narrower_frame_fills_null(db: Database) -> None:
    db.upsert("games", _sample(), keys=["game_pk"])
    narrower = pl.DataFrame({"game_pk": [5], "status": ["Final"]})
    db.upsert("games", narrower, keys=["game_pk"])
    result = db.query("SELECT home_score FROM games WHERE game_pk = 5")
    assert result["home_score"][0] is None


def test_upsert_composite_keys(db: Database) -> None:
    df = pl.DataFrame({"team_id": [1, 1], "season": [2024, 2025], "name": ["A", "B"]})
    db.upsert("teams", df, keys=["team_id", "season"])
    df2 = pl.DataFrame({"team_id": [1], "season": [2025], "name": ["B2"]})
    db.upsert("teams", df2, keys=["team_id", "season"])
    result = db.query("SELECT name FROM teams ORDER BY season")
    assert result["name"].to_list() == ["A", "B2"]
