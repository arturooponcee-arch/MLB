"""Tests de utilidades de DataFrames."""

import pandas as pd
import pytest

from mlb_quant.utils.frames import sanitize_column, sanitize_columns


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("K%", "k_pct"),
        ("BB%", "bb_pct"),
        ("wRC+", "wrc_plus"),
        ("HR/FB", "hr_fb"),
        ("Barrel%", "barrel_pct"),
        ("Hard Hit %", "hard_hit_pct"),
        ("IDfg", "idfg"),
        ("Season", "season"),
        ("1B", "x1b"),
        ("K-BB%", "k_bb_pct"),
        ("wOBA", "woba"),
    ],
)
def test_sanitize_column(original: str, expected: str) -> None:
    assert sanitize_column(original) == expected


def test_sanitize_columns_renames_all() -> None:
    df = pd.DataFrame({"K%": [0.25], "wRC+": [110], "Name": ["Aaron Judge"]})
    result = sanitize_columns(df)
    assert list(result.columns) == ["k_pct", "wrc_plus", "name"]


def test_sanitize_columns_rejects_collisions() -> None:
    df = pd.DataFrame([[1, 2]], columns=["K%", "K_pct"])
    with pytest.raises(ValueError, match="duplicadas"):
        sanitize_columns(df)


def test_sanitize_columns_does_not_mutate_original() -> None:
    df = pd.DataFrame({"K%": [0.25]})
    sanitize_columns(df)
    assert list(df.columns) == ["K%"]
