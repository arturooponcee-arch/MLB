"""Tests de la fuente Statcast (pybaseball mockeado)."""

from datetime import date

import pandas as pd
import pytest

from mlb_quant.ingestion.statcast import StatcastSource


class _StubStatcast(StatcastSource):
    """Sustituye la descarga real por un DataFrame fijo."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def _download(self, start_date: date, end_date: date) -> pd.DataFrame:
        return self._frame


def _pitches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 1, 2],
            "pitch_number": [1, 2, 1],
            "release_speed": [95.2, 88.1, 92.4],
        }
    )


def test_fetch_pitches_returns_rows() -> None:
    source = _StubStatcast(_pitches())
    df = source.fetch_pitches(date(2025, 7, 1), date(2025, 7, 1))
    assert len(df) == 3


def test_fetch_pitches_deduplicates_by_key() -> None:
    frame = pd.concat([_pitches(), _pitches()], ignore_index=True)
    source = _StubStatcast(frame)
    df = source.fetch_pitches(date(2025, 7, 1), date(2025, 7, 1))
    assert len(df) == 3


def test_fetch_pitches_empty_is_ok() -> None:
    source = _StubStatcast(pd.DataFrame())
    df = source.fetch_pitches(date(2026, 7, 15), date(2026, 7, 15))
    assert df.empty


def test_fetch_pitches_missing_keys_raises() -> None:
    source = _StubStatcast(pd.DataFrame({"release_speed": [95.0]}))
    with pytest.raises(ValueError, match="columnas clave"):
        source.fetch_pitches(date(2025, 7, 1), date(2025, 7, 1))
