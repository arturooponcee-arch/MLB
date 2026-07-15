"""Tests de la fuente The Odds API (sin red)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mlb_quant.ingestion.odds_api import ODDS_KEYS, ODDS_URL, OddsApiSource

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self._payload = payload
        self.headers = {"x-requests-remaining": "499"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, Any]]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        return _FakeResponse(self._payload)


@pytest.fixture
def odds_payload() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "odds_sample.json").read_text(encoding="utf-8"))


def _source(payload: list[dict[str, Any]]) -> tuple[OddsApiSource, _FakeSession]:
    session = _FakeSession(payload)
    source = OddsApiSource(
        api_key="test-key",
        session=session,  # type: ignore[arg-type]
        now=datetime(2026, 7, 15, 18, 30, tzinfo=UTC),
    )
    return source, session


def test_empty_api_key_raises() -> None:
    with pytest.raises(ValueError, match="ODDS_API_KEY"):
        OddsApiSource(api_key="")


def test_fetch_odds_parses_tidy_rows(odds_payload: list[dict[str, Any]]) -> None:
    source, _ = _source(odds_payload)
    df = source.fetch_odds()
    # evento 1: 2 casas x (2 h2h + 2 totals en DK, 2 h2h en FD) = 6; evento 2: 2 h2h.
    assert len(df) == 8
    assert df["event_id"].n_unique() == 2
    assert set(df["market"].unique().to_list()) == {"h2h", "totals"}
    assert df["snapshot_utc"].unique().to_list() == ["2026-07-15T18:30:00Z"]


def test_fetch_odds_point_only_for_totals(odds_payload: list[dict[str, Any]]) -> None:
    source, _ = _source(odds_payload)
    df = source.fetch_odds()
    assert df.filter(df["market"] == "h2h")["point"].null_count() == 6
    assert df.filter(df["market"] == "totals")["point"].to_list() == [8.5, 8.5]


def test_fetch_odds_request_params(odds_payload: list[dict[str, Any]]) -> None:
    source, session = _source(odds_payload)
    source.fetch_odds(markets=("h2h",), regions=("us", "eu"))
    call = session.calls[0]
    assert call["url"] == ODDS_URL
    assert call["params"]["apiKey"] == "test-key"
    assert call["params"]["markets"] == "h2h"
    assert call["params"]["regions"] == "us,eu"
    assert call["params"]["oddsFormat"] == "decimal"


def test_empty_payload_gives_empty_frame() -> None:
    source, _ = _source([])
    df = source.fetch_odds()
    assert df.is_empty()
    assert set(ODDS_KEYS) <= set(df.columns)


def test_rows_unique_by_upsert_keys(odds_payload: list[dict[str, Any]]) -> None:
    source, _ = _source(odds_payload)
    df = source.fetch_odds()
    assert df.select(list(ODDS_KEYS)).n_unique() == len(df)
