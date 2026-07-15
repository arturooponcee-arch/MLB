"""Tests del ensemble calibrado."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from mlb_quant.models.ensemble import CalibratedEnsembleWinModel


@pytest.fixture
def synthetic_features() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 1200
    strength = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "game_pk": np.arange(n),
            "game_date": [date(2024, 4, 1) + timedelta(days=i // 10) for i in range(n)],
            "season": [2024] * n,
            "home_team_id": [1] * n,
            "away_team_id": [2] * n,
            "home_score": rng.poisson(np.exp(1.55 + 0.25 * strength)),
            "away_score": rng.poisson(np.exp(1.45 - 0.25 * strength)),
            "strength_diff": strength,
        }
    )


def test_predictions_in_valid_range(synthetic_features: pl.DataFrame) -> None:
    model = CalibratedEnsembleWinModel().fit(synthetic_features)
    preds = model.predict_home_win(synthetic_features)
    assert ((preds["p_home_win"] >= 0.01) & (preds["p_home_win"] <= 0.99)).all()


def test_learns_direction_of_signal(synthetic_features: pl.DataFrame) -> None:
    model = CalibratedEnsembleWinModel().fit(synthetic_features)
    strong = synthetic_features.filter(pl.col("strength_diff") > 1.0)
    weak = synthetic_features.filter(pl.col("strength_diff") < -1.0)
    assert (
        model.predict_home_win(strong)["p_home_win"].mean()
        > model.predict_home_win(weak)["p_home_win"].mean()
    )


def test_requires_labels(synthetic_features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="home_score"):
        CalibratedEnsembleWinModel().fit(synthetic_features.drop("home_score"))


def test_too_few_games_raises(synthetic_features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="pocos juegos"):
        CalibratedEnsembleWinModel().fit(synthetic_features.head(60))


def test_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError, match="calibration_fraction"):
        CalibratedEnsembleWinModel(calibration_fraction=0.7)


def test_all_null_and_constant_columns_excluded(
    synthetic_features: pl.DataFrame,
) -> None:
    degenerate = synthetic_features.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("all_null"),
        pl.lit(1.0).alias("constant"),
    )
    model = CalibratedEnsembleWinModel().fit(degenerate)
    preds = model.predict_home_win(degenerate)
    assert len(preds) == len(degenerate)


def test_handles_missing_feature_columns(synthetic_features: pl.DataFrame) -> None:
    model = CalibratedEnsembleWinModel().fit(synthetic_features)
    preds = model.predict_home_win(synthetic_features.drop("strength_diff"))
    assert len(preds) == len(synthetic_features)
