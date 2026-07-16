"""Tests de las curvas de calibración (visualization.calibration)."""

import re

import numpy as np
import polars as pl

from mlb_quant.evaluation.metrics import calibration_table
from mlb_quant.visualization.calibration import calibration_svg, render_calibration_html


def _table(n_bins: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    return calibration_table(y, p, n_bins=n_bins)


def test_svg_well_formed_with_one_point_per_bin() -> None:
    table = _table(n_bins=5)
    svg = calibration_svg(table, "Temporada 2024")
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert svg.count("<circle") == len(table)
    assert "Temporada 2024" in svg


def test_html_embeds_one_svg_per_season() -> None:
    per_season = {2023: _table(), 2024: _table(), 2025: _table()}
    metrics = {
        season: {"brier": 0.24, "log_loss": 0.68, "n": 1500.0} for season in per_season
    }
    html = render_calibration_html(per_season, "p_home_ensemble", metrics)
    assert html.count("<svg") == 3
    assert "Temporada 2023" in html and "Temporada 2025" in html
    assert "p_home_ensemble" in html
    assert "0.2400" in html  # brier en el caption


def test_svg_radius_scales_with_n() -> None:
    table = pl.DataFrame(
        {
            "bin_low": [0.0, 0.5],
            "bin_high": [0.5, 1.0],
            "p_predicted": [0.25, 0.75],
            "p_observed": [0.24, 0.76],
            "n": [10, 1000],
        }
    )
    svg = calibration_svg(table, "radios")
    radii = [float(m) for m in re.findall(r'<circle[^>]*\br="([\d.]+)"', svg)]
    assert len(radii) == 2
    assert radii[1] > radii[0]
