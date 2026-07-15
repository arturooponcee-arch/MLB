"""Tests de la simulación Monte Carlo."""

import polars as pl
import pytest

from mlb_quant.models.markets import market_probabilities
from mlb_quant.simulations.monte_carlo import SimulationConfig, simulate_games


def _lambdas(pairs: list[tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": list(range(len(pairs))),
            "lambda_home": [p[0] for p in pairs],
            "lambda_away": [p[1] for p in pairs],
        }
    )


CFG = SimulationConfig(n_sims=20_000, seed=7)


def test_symmetric_game_is_even() -> None:
    result = simulate_games(_lambdas([(4.5, 4.5)]), CFG)
    assert result["sim_p_home_ml"][0] == pytest.approx(0.5, abs=0.02)


def test_no_ties_after_extras() -> None:
    result = simulate_games(_lambdas([(4.5, 4.5), (2.0, 2.0)]), CFG)
    total = result["sim_p_home_ml"] + result["sim_p_away_ml"]
    assert total.to_list() == pytest.approx([1.0, 1.0], abs=1e-12)


def test_agrees_with_analytic_grid() -> None:
    """Validación cruzada: MC y malla analítica deben coincidir en ML."""
    lambdas = _lambdas([(5.2, 3.8), (3.5, 4.9)])
    mc = simulate_games(lambdas, CFG)
    analytic = market_probabilities(lambdas)
    for i in range(2):
        assert mc["sim_p_home_ml"][i] == pytest.approx(
            analytic["p_home_ml"][i], abs=0.02
        )


def test_runline_below_moneyline_for_favorite() -> None:
    result = simulate_games(_lambdas([(5.5, 3.5)]), CFG)
    assert result["sim_p_home_runline"][0] < result["sim_p_home_ml"][0]


def test_extras_inflate_total_beyond_lambdas() -> None:
    result = simulate_games(_lambdas([(4.5, 4.5)]), CFG)
    assert result["sim_exp_total"][0] > 9.0  # 4.5+4.5 más carreras de extras


def test_f5_symmetric_game() -> None:
    result = simulate_games(_lambdas([(4.5, 4.5)]), CFG)
    p_tie = result["sim_p_f5_tie"][0]
    assert p_tie > 0.1  # empate F5 es común
    # Juego simétrico: victorias F5 repartidas por igual fuera del empate.
    assert result["sim_p_f5_home_ml"][0] == pytest.approx((1 - p_tie) / 2, abs=0.02)


def test_reproducible_with_seed() -> None:
    a = simulate_games(_lambdas([(4.8, 4.1)]), SimulationConfig(n_sims=5000, seed=99))
    b = simulate_games(_lambdas([(4.8, 4.1)]), SimulationConfig(n_sims=5000, seed=99))
    assert a["sim_p_home_ml"][0] == b["sim_p_home_ml"][0]


def test_quantiles_ordered() -> None:
    result = simulate_games(_lambdas([(4.5, 4.0)]), CFG)
    assert result["sim_total_p10"][0] < result["sim_total_p50"][0]
    assert result["sim_total_p50"][0] < result["sim_total_p90"][0]


def test_chunking_covers_all_games() -> None:
    lambdas = _lambdas([(4.0, 4.0)] * 7)
    result = simulate_games(lambdas, SimulationConfig(n_sims=1000, seed=1, chunk_size=3))
    assert result.height == 7
    assert result["game_pk"].to_list() == list(range(7))
