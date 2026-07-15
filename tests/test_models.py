"""Tests de los modelos Poisson y logístico sobre datos sintéticos."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mlb_quant.models.base import feature_columns, load_model, save_model
from mlb_quant.models.logistic import LogisticWinModel
from mlb_quant.models.poisson import PoissonRunsModel


@pytest.fixture
def synthetic_features() -> pl.DataFrame:
    """600 juegos: la feature 'strength_diff' gobierna carreras y victorias."""
    rng = np.random.default_rng(42)
    n = 600
    strength = rng.normal(0, 1, n)
    lambda_home = np.exp(1.55 + 0.25 * strength)
    lambda_away = np.exp(1.45 - 0.25 * strength)
    return pl.DataFrame(
        {
            "game_pk": np.arange(n),
            "game_date": [date(2024, 4, 1) + timedelta(days=i // 10) for i in range(n)],
            "season": [2024] * n,
            "home_team_id": [1] * n,
            "away_team_id": [2] * n,
            "home_score": rng.poisson(lambda_home),
            "away_score": rng.poisson(lambda_away),
            "strength_diff": strength,
            "noise": rng.normal(0, 1, n),
        }
    )


def test_feature_columns_excludes_meta(synthetic_features: pl.DataFrame) -> None:
    cols = feature_columns(synthetic_features)
    assert "strength_diff" in cols
    assert "home_score" not in cols
    assert "game_pk" not in cols


def test_poisson_learns_signal(synthetic_features: pl.DataFrame) -> None:
    model = PoissonRunsModel(alpha=0.1).fit(synthetic_features)
    strong = synthetic_features.filter(pl.col("strength_diff") > 1.0)
    weak = synthetic_features.filter(pl.col("strength_diff") < -1.0)
    assert (
        model.predict_lambdas(strong)["lambda_home"].mean()
        > model.predict_lambdas(weak)["lambda_home"].mean()
    )


def test_poisson_lambdas_positive(synthetic_features: pl.DataFrame) -> None:
    model = PoissonRunsModel().fit(synthetic_features)
    lambdas = model.predict_lambdas(synthetic_features)
    assert (lambdas["lambda_home"] > 0).all()
    assert (lambdas["lambda_away"] > 0).all()


def test_poisson_requires_labels(synthetic_features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="home_score"):
        PoissonRunsModel().fit(synthetic_features.drop("home_score"))


def test_poisson_handles_missing_feature_column(
    synthetic_features: pl.DataFrame,
) -> None:
    model = PoissonRunsModel().fit(synthetic_features)
    lambdas = model.predict_lambdas(synthetic_features.drop("noise"))
    assert len(lambdas) == len(synthetic_features)


def test_logistic_probabilities_valid(synthetic_features: pl.DataFrame) -> None:
    model = LogisticWinModel().fit(synthetic_features)
    preds = model.predict_home_win(synthetic_features)
    assert ((preds["p_home_win"] > 0) & (preds["p_home_win"] < 1)).all()


def test_coefficients_ranked_and_named(synthetic_features: pl.DataFrame) -> None:
    model = PoissonRunsModel(alpha=0.1).fit(synthetic_features)
    coefs = model.coefficients("home")
    assert set(coefs["feature"].to_list()) == {"strength_diff", "noise"}
    # La señal domina al ruido y con signo positivo para el local.
    top = coefs.row(0, named=True)
    assert top["feature"] == "strength_diff"
    assert top["coefficient"] > 0
    assert coefs["coefficient"].to_list() == sorted(
        coefs["coefficient"].to_list(), reverse=True
    )


def test_coefficients_invalid_side(synthetic_features: pl.DataFrame) -> None:
    model = PoissonRunsModel().fit(synthetic_features)
    with pytest.raises(ValueError, match="side"):
        model.coefficients("centro")


def test_coefficients_unfitted_raises() -> None:
    with pytest.raises(ValueError, match="entrenar"):
        PoissonRunsModel().coefficients("home")


def test_save_load_roundtrip(
    synthetic_features: pl.DataFrame, tmp_path: Path
) -> None:
    model = PoissonRunsModel().fit(synthetic_features)
    before = model.predict_lambdas(synthetic_features)["lambda_home"].to_list()
    save_model(model, tmp_path / "m.joblib")
    restored = load_model(tmp_path / "m.joblib")
    after = restored.predict_lambdas(synthetic_features)["lambda_home"].to_list()
    assert before == after
