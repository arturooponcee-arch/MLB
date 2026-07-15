"""Tests de la fuente de clima Open-Meteo (sin red, salvo integración)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mlb_quant.ingestion.weather import ARCHIVE_URL, FORECAST_URL, WeatherSource

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
def weather_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "weather_sample.json").read_text(encoding="utf-8"))


def _source(payload: dict[str, Any], today: date) -> tuple[WeatherSource, _FakeSession]:
    session = _FakeSession(payload)
    source = WeatherSource(session=session, today=today)  # type: ignore[arg-type]
    return source, session


def test_fetch_hourly_parses_rows(weather_payload: dict[str, Any]) -> None:
    source, _ = _source(weather_payload, today=date(2026, 7, 15))
    df = source.fetch_hourly(3313, 40.83, -73.93, date(2025, 7, 1), date(2025, 7, 1))
    assert len(df) == 3
    assert df["venue_id"].to_list() == [3313, 3313, 3313]
    assert df["temperature_c"][0] == 28.4
    assert df["wind_direction_deg"][2] == 240.0


def test_past_range_uses_archive(weather_payload: dict[str, Any]) -> None:
    source, session = _source(weather_payload, today=date(2026, 7, 15))
    source.fetch_hourly(1, 0.0, 0.0, date(2025, 7, 1), date(2025, 7, 2))
    assert session.calls[0]["url"] == ARCHIVE_URL


def test_range_including_today_uses_forecast(weather_payload: dict[str, Any]) -> None:
    source, session = _source(weather_payload, today=date(2025, 7, 1))
    source.fetch_hourly(1, 0.0, 0.0, date(2025, 7, 1), date(2025, 7, 1))
    assert session.calls[0]["url"] == FORECAST_URL


def test_empty_payload_gives_empty_frame() -> None:
    source, _ = _source({"hourly": {}}, today=date(2026, 7, 15))
    df = source.fetch_hourly(1, 0.0, 0.0, date(2025, 7, 1), date(2025, 7, 1))
    assert df.is_empty()
    assert "time_utc" in df.columns


@pytest.mark.integration
def test_fetch_hourly_real_api() -> None:
    source = WeatherSource()
    df = source.fetch_hourly(3313, 40.83, -73.93, date(2025, 7, 1), date(2025, 7, 1))
    assert len(df) == 24
    assert df["temperature_c"].null_count() == 0
