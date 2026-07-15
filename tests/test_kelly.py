"""Tests del criterio de Kelly (casos analíticos)."""

import pytest

from mlb_quant.bankroll.kelly import expected_value, kelly_fraction, recommend_stake


def test_expected_value_analytic() -> None:
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.20)


def test_kelly_fraction_analytic() -> None:
    # p=0.6, cuota 2.0 -> b=1 -> f = (0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    # p=0.5, cuota 3.0 -> b=2 -> f = (1.0 - 0.5)/2 = 0.25
    assert kelly_fraction(0.5, 3.0) == pytest.approx(0.25)


def test_no_edge_gives_zero_stake() -> None:
    assert kelly_fraction(0.45, 2.0) == 0.0
    rec = recommend_stake(1000, 0.45, 2.0)
    assert rec.stake == 0.0
    assert rec.ev < 0


def test_half_kelly_and_cap() -> None:
    rec = recommend_stake(1000, 0.6, 2.0, kelly_multiplier=0.5, max_fraction=0.05)
    # Half Kelly = 0.1 -> recortado al tope 0.05 -> 50.
    assert rec.capped
    assert rec.stake == pytest.approx(50.0)
    relaxed = recommend_stake(1000, 0.6, 2.0, kelly_multiplier=0.5, max_fraction=0.5)
    assert not relaxed.capped
    assert relaxed.stake == pytest.approx(100.0)


def test_fair_odds() -> None:
    rec = recommend_stake(1000, 0.5, 2.2)
    assert rec.fair_odds == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("p", "odds"),
    [(0.0, 2.0), (1.0, 2.0), (0.5, 1.0), (0.5, 0.9), (-0.1, 2.0)],
)
def test_invalid_inputs_raise(p: float, odds: float) -> None:
    with pytest.raises(ValueError):
        kelly_fraction(p, odds)


def test_invalid_bankroll_raises() -> None:
    with pytest.raises(ValueError, match="bankroll"):
        recommend_stake(0, 0.6, 2.0)
