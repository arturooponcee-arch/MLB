"""Tests del cliente MLB Stats API (sin red, salvo integración)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mlb_quant.ingestion.mlb_stats_api import MlbStatsApi

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        return _FakeResponse(self._payload)


@pytest.fixture
def schedule_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "schedule_sample.json").read_text(encoding="utf-8"))


def test_fetch_schedule_parses_games(schedule_payload: dict[str, Any]) -> None:
    session = _FakeSession(schedule_payload)
    api = MlbStatsApi(session=session)  # type: ignore[arg-type]

    df = api.fetch_schedule(date(2025, 7, 1), date(2025, 7, 1))

    assert len(df) == 2
    final = df.filter(df["game_pk"] == 777001)
    assert final["home_team_name"][0] == "New York Yankees"
    assert final["home_score"][0] == 5
    assert final["home_probable_pitcher_name"][0] == "Gerrit Cole"
    assert final["game_date"][0] == date(2025, 7, 1)


def test_fetch_schedule_handles_missing_fields(schedule_payload: dict[str, Any]) -> None:
    """Juego programado: sin score ni probables. No debe fallar."""
    session = _FakeSession(schedule_payload)
    api = MlbStatsApi(session=session)  # type: ignore[arg-type]

    df = api.fetch_schedule(date(2025, 7, 1), date(2025, 7, 1))

    scheduled = df.filter(df["game_pk"] == 777002)
    assert scheduled["home_score"][0] is None
    assert scheduled["home_probable_pitcher_id"][0] is None
    assert scheduled["status"][0] == "Scheduled"


def test_fetch_schedule_requests_probable_pitcher_hydrate(
    schedule_payload: dict[str, Any],
) -> None:
    session = _FakeSession(schedule_payload)
    api = MlbStatsApi(session=session)  # type: ignore[arg-type]

    api.fetch_schedule(date(2025, 7, 1), date(2025, 7, 2))

    params = session.calls[0]["params"]
    assert params["hydrate"] == "probablePitcher"
    assert params["startDate"] == "2025-07-01"
    assert params["endDate"] == "2025-07-02"


def test_fetch_schedule_empty_dates() -> None:
    session = _FakeSession({"dates": []})
    api = MlbStatsApi(session=session)  # type: ignore[arg-type]

    df = api.fetch_schedule(date(2026, 7, 15), date(2026, 7, 15))

    assert df.is_empty()
    assert "game_pk" in df.columns


@pytest.mark.integration
def test_fetch_schedule_real_api() -> None:
    """Contra el servidor real: 2025-07-01 tuvo juegos de temporada regular."""
    api = MlbStatsApi()
    df = api.fetch_schedule(date(2025, 7, 1), date(2025, 7, 1))
    assert len(df) > 0
    assert df["game_pk"].is_duplicated().sum() == 0


@pytest.mark.integration
def test_fetch_teams_real_api() -> None:
    api = MlbStatsApi()
    df = api.fetch_teams(2025)
    assert len(df) == 30
