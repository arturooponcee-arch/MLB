"""Tests de la fuente Baseball Savant (sin red, salvo integración)."""

from typing import Any

import pytest

from mlb_quant.ingestion.savant import SavantLeaderboards

_EXPECTED_CSV = (
    '"last_name, first_name","player_id","year","pa","ba","est_ba","woba","est_woba"\n'
    '"Judge, Aaron",592450,2025,704,0.322,0.310,".458",".445"\n'
    '"Soto, Juan",665742,2025,713,0.288,0.295,".420",".431"\n'
)

_CUSTOM_CSV = (
    '"last_name, first_name","player_id","year","pa","k_percent","xwoba","barrel_batted_rate"\n'
    '"Judge, Aaron",592450,2025,704,24.5,".445",26.6\n'
)


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self.content = body.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params})
        return _FakeResponse(self._body)


def test_fetch_expected_stats_parses_and_tags() -> None:
    session = _FakeSession(_EXPECTED_CSV)
    source = SavantLeaderboards(session=session)  # type: ignore[arg-type]

    df = source.fetch_expected_stats(2025, "batter")

    assert len(df) == 2
    assert "name" in df.columns
    assert df["player_type"].unique().tolist() == ["batter"]
    judge = df[df["player_id"] == 592450]
    assert judge["est_woba"].iloc[0] == pytest.approx(0.445)


def test_fetch_statcast_metrics_sanitizes() -> None:
    session = _FakeSession(_CUSTOM_CSV)
    source = SavantLeaderboards(session=session)  # type: ignore[arg-type]

    df = source.fetch_statcast_metrics(2025, "pitcher")

    assert "k_percent" in df.columns
    assert df["player_type"].iloc[0] == "pitcher"
    assert "selections" in session.calls[0]["params"]


def test_missing_keys_raises() -> None:
    session = _FakeSession('"nombre"\n"X"\n')
    source = SavantLeaderboards(session=session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clave"):
        source.fetch_expected_stats(2025, "batter")


def test_deduplicates_by_key() -> None:
    duplicated = _EXPECTED_CSV + '"Judge, Aaron",592450,2025,704,0.322,0.310,".458",".445"\n'
    source = SavantLeaderboards(session=_FakeSession(duplicated))  # type: ignore[arg-type]

    df = source.fetch_expected_stats(2025, "batter")

    assert len(df) == 2


@pytest.mark.integration
def test_fetch_expected_stats_real_api() -> None:
    source = SavantLeaderboards()
    df = source.fetch_expected_stats(2025, "batter", min_pa=100)
    assert len(df) > 100
    assert df["est_woba"].notna().all()


@pytest.mark.integration
def test_fetch_statcast_metrics_real_api() -> None:
    source = SavantLeaderboards()
    df = source.fetch_statcast_metrics(2025, "pitcher", min_pa=100)
    assert len(df) > 100
    assert "barrel_batted_rate" in df.columns
