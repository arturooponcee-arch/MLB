"""Métricas de evaluación de probabilidades."""

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def probability_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Métricas estándar para probabilidades binarias.

    Args:
        y_true: Resultados observados (0/1).
        p: Probabilidades predichas.

    Returns:
        ``log_loss``, ``brier``, ``roc_auc``, ``accuracy``, ``n``.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(((p >= 0.5) == y_true).mean()),
        "n": float(len(y_true)),
    }


def calibration_table(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> pl.DataFrame:
    """Tabla de calibración: probabilidad predicha vs frecuencia observada.

    Args:
        y_true: Resultados observados (0/1).
        p: Probabilidades predichas.
        n_bins: Número de bins uniformes en [0, 1].

    Returns:
        Por bin: rango, probabilidad media predicha, frecuencia observada
        y número de casos. Bins vacíos se omiten.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "p_predicted": float(p[mask].mean()),
                "p_observed": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return pl.DataFrame(rows)
