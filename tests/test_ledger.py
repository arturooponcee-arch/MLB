"""Tests del ledger de apuestas y CLV (betting.ledger)."""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from mlb_quant.betting.ledger import (
    BETS_KEYS,
    BETS_SCHEMA,
    closing_odds,
    create_bet,
    ledger_summary,
    settle_bets,
)

NOW = datetime(2026, 7, 15, 15, 0, tzinfo=UTC)

_GAME = {
    "game_pk": 1001,
    "game_date": date(2026, 7, 15),
    "game_datetime_utc": "2026-07-15T23:05:00Z",
    "home_team_name": "Boston Red Sox",
    "away_team_name": "New York Yankees",
    "status": "Scheduled",
}


def _games(home_score: int | None = None, away_score: int | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1001],
            "game_date": [date(2026, 7, 15)],
            "game_datetime_utc": ["2026-07-15T23:05:00Z"],
            "home_team_name": ["Boston Red Sox"],
            "away_team_name": ["New York Yankees"],
            "status": ["Final" if home_score is not None else "Scheduled"],
            "home_score": [home_score],
            "away_score": [away_score],
        }
    )


def _snapshot_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "snapshot_utc": "2026-07-15T20:00:00Z",
        "event_id": "manual-1001",
        "commence_time_utc": "2026-07-15T23:05:00Z",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "book": "caliente",
        "market": "h2h",
        "outcome": "Boston Red Sox",
        "point": None,
        "decimal_odds": 2.0,
        "last_update_utc": "2026-07-15T20:00:00Z",
    }
    base.update(overrides)
    return base


def test_create_bet_schema_and_fields() -> None:
    bet = create_bet(_GAME, "home", 2.1, 50.0, "caliente", p_model=0.52, now=NOW)
    assert bet.schema == pl.Schema(BETS_SCHEMA)
    row = bet.row(0, named=True)
    assert row["team_name"] == "Boston Red Sox"
    assert row["status"] == "open"
    assert row["market"] == "moneyline"
    assert len(row["bet_id"]) == 12
    assert bet.select(list(BETS_KEYS)).n_unique() == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("selection", "over", "selection"), ("odds", 0.9, "cuota"), ("stake", 0.0, "stake")],
)
def test_create_bet_validates(field: str, value: float | str, match: str) -> None:
    kwargs = {"selection": "home", "decimal_odds": 2.0, "stake": 10.0}
    kwargs[field if field != "odds" else "decimal_odds"] = value
    with pytest.raises(ValueError, match=match):
        create_bet(_GAME, kwargs["selection"], kwargs["decimal_odds"], kwargs["stake"], "b")


def test_settle_win_loss_profit() -> None:
    home_bet = create_bet(_GAME, "home", 2.5, 100.0, "b", now=NOW)
    away_bet = create_bet(_GAME, "away", 1.8, 40.0, "b", now=NOW)
    bets = pl.concat([home_bet, away_bet])
    settled = settle_bets(bets, _games(home_score=5, away_score=3), now=NOW)
    assert len(settled) == 2
    home_row = settled.filter(pl.col("selection") == "home").row(0, named=True)
    away_row = settled.filter(pl.col("selection") == "away").row(0, named=True)
    assert home_row["status"] == "won"
    assert home_row["profit"] == pytest.approx(100.0 * 1.5)
    assert away_row["status"] == "lost"
    assert away_row["profit"] == pytest.approx(-40.0)


def test_settle_tie_is_void() -> None:
    bets = create_bet(_GAME, "home", 2.0, 10.0, "b", now=NOW)
    settled = settle_bets(bets, _games(home_score=2, away_score=2), now=NOW)
    row = settled.row(0, named=True)
    assert row["status"] == "void"
    assert row["profit"] == 0.0


def test_settle_skips_unfinished_and_settled() -> None:
    open_bet = create_bet(_GAME, "home", 2.0, 10.0, "b", now=NOW)
    already = open_bet.with_columns(pl.lit("won").alias("status"))
    assert settle_bets(open_bet, _games(), now=NOW).is_empty()  # juego sin terminar
    assert settle_bets(already, _games(5, 3), now=NOW).is_empty()  # ya liquidada


def test_closing_prefers_last_pre_commence_snapshot() -> None:
    bet = create_bet(_GAME, "home", 2.2, 10.0, "caliente", now=NOW)
    snapshots = pl.DataFrame(
        [
            _snapshot_row(snapshot_utc="2026-07-15T12:00:00Z", decimal_odds=2.3),
            _snapshot_row(snapshot_utc="2026-07-15T22:30:00Z", decimal_odds=2.0),
            # Post-commence: no cuenta como cierre.
            _snapshot_row(snapshot_utc="2026-07-15T23:30:00Z", decimal_odds=1.5),
        ]
    )
    result = closing_odds(bet, snapshots, _games())
    row = result.row(0, named=True)
    assert row["closing_odds"] == pytest.approx(2.0)
    assert row["clv_pct"] == pytest.approx(2.2 / 2.0 - 1.0)


def test_closing_prefers_same_book_else_median() -> None:
    bet = create_bet(_GAME, "home", 2.2, 10.0, "caliente", now=NOW)
    snapshots = pl.DataFrame(
        [
            _snapshot_row(book="caliente", decimal_odds=2.1),
            _snapshot_row(book="bet365", decimal_odds=1.9),
            _snapshot_row(book="pinnacle", decimal_odds=2.5),
        ]
    )
    assert closing_odds(bet, snapshots, _games())["closing_odds"][0] == pytest.approx(2.1)

    other_books = snapshots.filter(pl.col("book") != "caliente")
    assert closing_odds(bet, other_books, _games())["closing_odds"][0] == pytest.approx(
        (1.9 + 2.5) / 2
    )


def test_closing_null_without_pre_game_snapshot() -> None:
    bet = create_bet(_GAME, "home", 2.2, 10.0, "b", now=NOW)
    post_only = pl.DataFrame([_snapshot_row(snapshot_utc="2026-07-16T01:00:00Z")])
    result = closing_odds(bet, post_only, _games())
    assert result["closing_odds"][0] is None
    assert result["clv_pct"][0] is None


def test_closing_matches_by_team_names_fallback() -> None:
    bet = create_bet(_GAME, "away", 2.0, 10.0, "b", now=NOW)
    api_snapshot = pl.DataFrame(
        [
            _snapshot_row(
                event_id="evt-api-xyz",
                outcome="New York Yankees",
                decimal_odds=1.85,
                home_team="  BOSTON RED SOX ",
            )
        ]
    )
    result = closing_odds(bet, api_snapshot, _games())
    assert result["closing_odds"][0] == pytest.approx(1.85)


def test_ledger_summary_hand_computed() -> None:
    won = create_bet(_GAME, "home", 2.0, 100.0, "b", now=NOW).with_columns(
        pl.lit("won").alias("status"),
        pl.lit(100.0).alias("profit"),
        pl.lit(0.05).alias("clv_pct"),
    )
    lost = create_bet(_GAME, "away", 1.8, 50.0, "b", now=NOW).with_columns(
        pl.lit("lost").alias("status"),
        pl.lit(-50.0).alias("profit"),
        pl.lit(-0.01).alias("clv_pct"),
    )
    still_open = create_bet(_GAME, "home", 2.0, 25.0, "b", now=NOW)
    summary = ledger_summary(pl.concat([won, lost, still_open]))
    assert summary["n"] == 3
    assert summary["n_open"] == 1
    assert summary["staked"] == pytest.approx(150.0)
    assert summary["profit"] == pytest.approx(50.0)
    assert summary["roi"] == pytest.approx(50.0 / 150.0)
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["clv_mean"] == pytest.approx(0.02)
    assert summary["n_clv"] == 2


def test_bets_roundtrip_in_duckdb(tmp_path: Path) -> None:
    from mlb_quant.db import Database

    db = Database(tmp_path / "ledger.duckdb")
    bet = create_bet(_GAME, "home", 2.0, 10.0, "b", now=NOW)
    db.upsert("bets", bet, keys=list(BETS_KEYS))
    stored = db.query("SELECT * FROM bets")
    assert len(stored) == 1
    assert stored["status"][0] == "open"
    # Upsert de la misma fila liquidada no duplica.
    settled = settle_bets(stored, _games(5, 3), now=NOW)
    db.upsert("bets", settled, keys=list(BETS_KEYS))
    assert len(db.query("SELECT * FROM bets")) == 1
    assert db.query("SELECT status FROM bets")["status"][0] == "won"
