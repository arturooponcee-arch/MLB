"""Tests de la fuente Lahman (descarga y pyreadr mockeados)."""

from pathlib import Path

import pandas as pd
import pytest

from mlb_quant.ingestion.lahman import LAHMAN_TABLES, LahmanSource


class _StubLahman(LahmanSource):
    """Sustituye la lectura del tarball por un DataFrame fijo."""

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__(cache_dir=Path("unused"))
        self._frame = frame

    def _load_rdata(self, rdata_name: str) -> pd.DataFrame:
        self.requested = rdata_name
        return self._frame


def test_fetch_table_sanitizes_columns() -> None:
    frame = pd.DataFrame({"playerID": ["ruthba01"], "yearID": [1927], "stint": [1]})
    source = _StubLahman(frame)
    df = source.fetch_table("batting")
    assert list(df.columns) == ["playerid", "yearid", "stint"]
    assert source.requested == "Batting"


def test_fetch_table_unknown_raises() -> None:
    source = _StubLahman(pd.DataFrame())
    with pytest.raises(ValueError, match="no soportada"):
        source.fetch_table("salaries")


def test_fetch_table_missing_keys_raises() -> None:
    source = _StubLahman(pd.DataFrame({"otra": [1]}))
    with pytest.raises(ValueError, match="clave"):
        source.fetch_table("teams")


def test_fetch_table_deduplicates() -> None:
    frame = pd.DataFrame(
        {"yearID": [2024, 2024], "teamID": ["NYA", "NYA"], "W": [94, 94]}
    )
    source = _StubLahman(frame)
    assert len(source.fetch_table("teams")) == 1


def test_all_specs_have_table_keys_and_rdata() -> None:
    for spec in LAHMAN_TABLES.values():
        assert spec.db_table.startswith("lahman_")
        assert spec.keys
        assert spec.rdata_name


def test_resolve_version_fallback_without_network(tmp_path: Path) -> None:
    class _FailingSession:
        def get(self, url: str, timeout: float) -> None:
            import requests

            raise requests.ConnectionError("sin red")

    source = LahmanSource(cache_dir=tmp_path, session=_FailingSession())  # type: ignore[arg-type]
    assert source._resolve_version() == "14.0-0"
