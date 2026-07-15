"""Tests de la derivación de mercados desde tasas Poisson."""

import numpy as np
import polars as pl
import pytest

from mlb_quant.models.markets import market_probabilities, poisson_pmf_matrix


def _lambdas(lh: float, la: float) -> pl.DataFrame:
    return pl.DataFrame({"game_pk": [1], "lambda_home": [lh], "lambda_away": [la]})


def test_pmf_rows_sum_to_one() -> None:
    pmf = poisson_pmf_matrix(np.array([2.0, 4.5, 9.0]))
    assert np.allclose(pmf.sum(axis=1), 1.0)


def test_pmf_matches_analytic() -> None:
    pmf = poisson_pmf_matrix(np.array([3.0]), max_runs=30)
    assert pmf[0, 0] == pytest.approx(np.exp(-3.0), rel=1e-9)
    assert pmf[0, 3] == pytest.approx(np.exp(-3.0) * 27 / 6, rel=1e-6)


def test_equal_lambdas_give_even_moneyline() -> None:
    result = market_probabilities(_lambdas(4.5, 4.5))
    assert result["p_home_ml"][0] == pytest.approx(0.5, abs=1e-9)
    assert result["p_away_ml"][0] == pytest.approx(0.5, abs=1e-9)


def test_stronger_home_is_favorite() -> None:
    result = market_probabilities(_lambdas(5.5, 3.5))
    assert result["p_home_ml"][0] > 0.6
    assert result["p_home_runline"][0] < result["p_home_ml"][0]


def test_over_under_complementary_and_monotonic() -> None:
    low = market_probabilities(_lambdas(3.0, 3.0), total_line=8.5)
    high = market_probabilities(_lambdas(6.0, 6.0), total_line=8.5)
    assert low["p_over"][0] + low["p_under"][0] == pytest.approx(1.0)
    assert high["p_over"][0] > low["p_over"][0]


def test_expected_values_match_lambdas() -> None:
    result = market_probabilities(_lambdas(4.2, 3.8))
    assert result["exp_home_runs"][0] == pytest.approx(4.2)
    assert result["exp_total"][0] == pytest.approx(8.0)
