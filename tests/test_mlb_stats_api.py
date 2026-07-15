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


def test_fetch_schedule_deduplicates_postponed_games() -> None:
    """Juego pospuesto: mismo game_pk en dos fechas. Gana la entrada Final."""
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 555,
                        "season": "2024",
                        "officialDate": "2024-04-02",
                        "gameDate": "2024-04-02T23:10:00Z",
                        "status": {"detailedState": "Postponed"},
                        "teams": {"home": {}, "away": {}},
                    }
                ]
            },
            {
                "games": [
                    {
                        "gamePk": 555,
                        "season": "2024",
                        "officialDate": "2024-04-04",
                        "gameDate": "2024-04-04T16:10:00Z",
                        "status": {"detailedState": "Final"},
                        "teams": {"home": {"score": 3}, "away": {"score": 1}},
                    }
                ]
            },
        ]
    }
    api = MlbStatsApi(session=_FakeSession(payload))  # type: ignore[arg-type]

    df = api.fetch_schedule(date(2024, 4, 2), date(2024, 4, 4))

    assert len(df) == 1
    assert df["status"][0] == "Final"
    assert df["game_date"][0] == date(2024, 4, 4)


def test_fetch_schedule_empty_dates() -> None:
    session = _FakeSession({"dates": []})
    api = MlbStatsApi(session=session)  # type: ignore[arg-type]

    df = api.fetch_schedule(date(2026, 7, 15), date(2026, 7, 15))

    assert df.is_empty()
    assert "game_pk" in df.columns


@pytest.fixture
def venues_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "venues_sample.json").read_text(encoding="utf-8"))


@pytest.fixture
def boxscore_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "boxscore_sample.json").read_text(encoding="utf-8"))


def test_fetch_venues_parses_full_venue(venues_payload: dict[str, Any]) -> None:
    api = MlbStatsApi(session=_FakeSession(venues_payload))  # type: ignore[arg-type]

    df = api.fetch_venues(2025)

    yankee = df.filter(df["venue_id"] == 3313)
    assert yankee["latitude"][0] == pytest.approx(40.8292, abs=1e-3)
    assert yankee["azimuth_angle"][0] == pytest.approx(75.38)
    assert yankee["elevation"][0] == 55.0
    assert yankee["roof_type"][0] == "Open"
    assert yankee["center"][0] == 408


def test_fetch_venues_handles_missing_fieldinfo(venues_payload: dict[str, Any]) -> None:
    api = MlbStatsApi(session=_FakeSession(venues_payload))  # type: ignore[arg-type]

    df = api.fetch_venues(2025)

    bare = df.filter(df["venue_id"] == 9999)
    assert bare["latitude"][0] is None
    assert bare["roof_type"][0] is None
    assert bare["city"][0] == "Nowhere"


def test_fetch_boxscore_lineups(boxscore_payload: dict[str, Any]) -> None:
    api = MlbStatsApi(session=_FakeSession(boxscore_payload))  # type: ignore[arg-type]

    lineups, _ = api.fetch_boxscore(777001)

    assert len(lineups) == 4  # 3 titulares + 1 emergente; banca y pitchers fuera
    judge = lineups.filter(lineups["player_id"] == 592450)
    assert judge["batting_order"][0] == 1
    assert judge["is_starting_lineup"][0] is True
    pinch = lineups.filter(lineups["player_id"] == 700100)
    assert pinch["batting_order"][0] == 2
    assert pinch["is_substitute"][0] is True


def test_fetch_boxscore_pitching_lines(boxscore_payload: dict[str, Any]) -> None:
    api = MlbStatsApi(session=_FakeSession(boxscore_payload))  # type: ignore[arg-type]

    _, pitching = api.fetch_boxscore(777001)

    assert len(pitching) == 3
    cole = pitching.filter(pitching["player_id"] == 543037)
    assert cole["is_starter"][0] is True
    assert cole["pitches_thrown"][0] == 95
    assert cole["outs_recorded"][0] == 18
    relief = pitching.filter(pitching["player_id"] == 660001)
    assert relief["is_starter"][0] is False
    bello = pitching.filter(pitching["player_id"] == 678394)
    assert bello["is_starter"][0] is True
    assert bello["team_id"][0] == 111


@pytest.mark.integration
def test_fetch_venues_real_api() -> None:
    api = MlbStatsApi()
    df = api.fetch_venues(2025)
    assert len(df) >= 30
    assert df["latitude"].null_count() < len(df)


@pytest.mark.integration
def test_fetch_boxscore_real_api() -> None:
    api = MlbStatsApi()
    lineups, pitching = api.fetch_boxscore(777283)
    assert len(lineups) >= 18
    assert len(pitching) >= 2
    assert pitching.filter(pitching["is_starter"]).height == 2


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
