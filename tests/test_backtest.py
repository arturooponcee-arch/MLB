"""Tests del backtest walk-forward."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from mlb_quant.evaluation.backtest import ENSEMBLE_CONFIGS, walk_forward
from mlb_quant.evaluation.metrics import calibration_table, probability_metrics


@pytest.fixture
def season_features() -> pl.DataFrame:
    """~1.000 juegos abril-agosto con señal real en strength_diff."""
    rng = np.random.default_rng(7)
    n = 1000
    strength = rng.normal(0, 1, n)
    dates = [date(2024, 4, 1) + timedelta(days=i // 8) for i in range(n)]
    return pl.DataFrame(
        {
            "game_pk": np.arange(n),
            "game_date": dates,
            "season": [2024] * n,
            "home_team_id": [1] * n,
            "away_team_id": [2] * n,
            "home_score": rng.poisson(np.exp(1.55 + 0.25 * strength)),
            "away_score": rng.poisson(np.exp(1.45 - 0.25 * strength)),
            "strength_diff": strength,
        }
    )


def test_walk_forward_produces_predictions(season_features: pl.DataFrame) -> None:
    result = walk_forward(
        season_features, date(2024, 6, 1), date(2024, 8, 31), min_train_games=300
    )
    assert result.predictions.height > 0
    assert "moneyline_poisson" in result.metrics
    assert result.metrics["moneyline_poisson"]["log_loss"] < 0.75


def test_walk_forward_never_predicts_training_period(
    season_features: pl.DataFrame,
) -> None:
    result = walk_forward(
        season_features, date(2024, 6, 1), date(2024, 8, 31), min_train_games=300
    )
    assert result.predictions["game_date"].min() >= date(2024, 6, 1)


def test_walk_forward_insufficient_history_raises(
    season_features: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="historia suficiente"):
        walk_forward(
            season_features, date(2024, 4, 1), date(2024, 4, 30), min_train_games=5000
        )


def test_walk_forward_compares_ensemble_heads(season_features: pl.DataFrame) -> None:
    result = walk_forward(
        season_features,
        date(2024, 6, 1),
        date(2024, 8, 31),
        min_train_games=300,
        ensemble_configs=ENSEMBLE_CONFIGS,
    )
    for name in ENSEMBLE_CONFIGS:
        column = f"p_home_ens_{name}"
        assert column in result.predictions.columns
        assert f"moneyline_{name}" in result.metrics
        p = result.predictions[column]
        assert ((p >= 0.01) & (p <= 0.99)).all()
    # La columna histórica sigue presente (config default del constructor).
    assert "p_home_ensemble" in result.predictions.columns
    # avg_platt replica la config default: mismas probabilidades.
    assert (
        (
            result.predictions["p_home_ensemble"]
            - result.predictions["p_home_ens_avg_platt"]
        )
        .abs()
        .max()
        < 1e-12
    )


def test_probability_metrics_perfect_predictions() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.99, 0.01, 0.99, 0.01])
    metrics = probability_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] < 0.05


def test_calibration_table_recovers_frequencies() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    table = calibration_table(y, p, n_bins=5)
    for row in table.iter_rows(named=True):
        assert row["p_observed"] == pytest.approx(row["p_predicted"], abs=0.03)
