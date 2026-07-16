"""Tests de los calibradores de probabilidad (models.calibration)."""

import numpy as np
import polars as pl
import pytest

from mlb_quant.evaluation.metrics import calibration_table
from mlb_quant.models.calibration import (
    BetaCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    make_calibrator,
)

_MIN_BIN_N = 200


def _distorted_sample(n: int = 20_000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad verdadera q ~ U(0.1, 0.9); score crudo distorsionado p = q^2."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(0, 1, n) < q).astype(int)
    return q**2, y


@pytest.mark.parametrize("kind", ["isotonic", "platt", "beta"])
def test_calibrator_inverts_known_distortion(kind: str) -> None:
    p_raw, y = _distorted_sample()
    calibrator = make_calibrator(kind)  # type: ignore[arg-type]
    calibrated = calibrator.fit(p_raw, y).predict(p_raw)
    table = calibration_table(y, calibrated, n_bins=8)
    # En cada bin poblado, lo predicho debe acercarse a lo observado.
    for row in table.filter(pl.col("n") >= _MIN_BIN_N).iter_rows(named=True):
        assert row["p_predicted"] == pytest.approx(row["p_observed"], abs=0.05)


def test_raw_score_is_miscalibrated_baseline() -> None:
    # Confirma que la distorsión de prueba es real: sin calibrar falla.
    p_raw, y = _distorted_sample()
    table = calibration_table(y, p_raw, n_bins=8)
    worst = max(
        abs(row["p_predicted"] - row["p_observed"])
        for row in table.filter(pl.col("n") >= _MIN_BIN_N).iter_rows(named=True)
    )
    assert worst > 0.10


def test_identity_clips() -> None:
    calibrator = IdentityCalibrator().fit(np.array([0.5]), np.array([1]))
    out = calibrator.predict(np.array([0.0, 0.5, 1.0]))
    assert out.tolist() == [0.01, 0.5, 0.99]


def test_outputs_within_bounds() -> None:
    p_raw, y = _distorted_sample(n=2000)
    for calibrator in (IsotonicCalibrator(), PlattCalibrator(), BetaCalibrator()):
        out = calibrator.fit(p_raw, y).predict(np.array([0.001, 0.5, 0.999]))
        assert (out >= 0.01).all() and (out <= 0.99).all()


def test_beta_recovers_identity_when_already_calibrated() -> None:
    # Si p ya está calibrada, beta debe aprender un mapa ~identidad.
    rng = np.random.default_rng(9)
    q = rng.uniform(0.05, 0.95, 30_000)
    y = (rng.uniform(0, 1, len(q)) < q).astype(int)
    calibrated = BetaCalibrator().fit(q, y).predict(q)
    assert float(np.mean(np.abs(calibrated - q))) < 0.02


def test_make_calibrator_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        make_calibrator("bogus")  # type: ignore[arg-type]
