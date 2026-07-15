"""Derivación de mercados desde las tasas Poisson.

Construye la distribución conjunta de carreras (independencia entre
lados) y de ahí calcula moneyline, run line, totales y carreras
esperadas por equipo.

Aproximaciones documentadas:
- Empates tras 9 innings se reparten proporcionalmente a la fuerza
  relativa (equivale a renormalizar sobre resultados sin empate).
- El run line ignora la dinámica de extra innings (los empates suelen
  resolverse por 1 carrera, así que no cubren el -1.5). La simulación
  Monte Carlo (fase 5) refina esto.
"""

import numpy as np
import polars as pl

#: Máximo de carreras consideradas por lado en la malla.
MAX_RUNS = 30


def poisson_pmf_matrix(lambdas: np.ndarray, max_runs: int = MAX_RUNS) -> np.ndarray:
    """PMF Poisson truncada y renormalizada por fila.

    Usa la recurrencia ``pmf[k] = pmf[k-1] * lambda / k`` (estable, sin
    factoriales).

    Args:
        lambdas: Tasas, forma ``(n,)``.
        max_runs: Último valor de carreras incluido.

    Returns:
        Matriz ``(n, max_runs + 1)`` con filas que suman 1.
    """
    n = len(lambdas)
    pmf = np.zeros((n, max_runs + 1), dtype=np.float64)
    pmf[:, 0] = np.exp(-lambdas)
    for k in range(1, max_runs + 1):
        pmf[:, k] = pmf[:, k - 1] * lambdas / k
    return pmf / pmf.sum(axis=1, keepdims=True)


def market_probabilities(
    lambdas: pl.DataFrame,
    total_line: float = 8.5,
    run_line: float = 1.5,
    max_runs: int = MAX_RUNS,
) -> pl.DataFrame:
    """Calcula los mercados de equipo desde las tasas por juego.

    Args:
        lambdas: ``game_pk``, ``lambda_home``, ``lambda_away``.
        total_line: Línea de over/under.
        run_line: Hándicap del favorito local (típico 1.5).
        max_runs: Truncamiento de la malla de carreras.

    Returns:
        Por juego: ``p_home_ml``, ``p_away_ml``, ``p_over``, ``p_under``,
        ``p_home_runline`` (local -run_line), ``p_away_runline``,
        ``exp_home_runs``, ``exp_away_runs``, ``exp_total``.
    """
    lh = lambdas["lambda_home"].to_numpy()
    la = lambdas["lambda_away"].to_numpy()
    pmf_h = poisson_pmf_matrix(lh, max_runs)  # (n, R)
    pmf_a = poisson_pmf_matrix(la, max_runs)

    runs = np.arange(max_runs + 1)
    diff = runs[:, None] - runs[None, :]  # h - a, forma (R, R)
    total = runs[:, None] + runs[None, :]

    joint = pmf_h[:, :, None] * pmf_a[:, None, :]  # (n, R, R)

    p_home_raw = joint[:, diff > 0].sum(axis=1)
    p_away_raw = joint[:, diff < 0].sum(axis=1)
    decided = p_home_raw + p_away_raw
    p_home_ml = p_home_raw / decided
    p_over = joint[:, total > total_line].sum(axis=1)
    p_home_rl = joint[:, diff > run_line].sum(axis=1)
    p_away_rl = 1.0 - p_home_rl

    return pl.DataFrame(
        {
            "game_pk": lambdas["game_pk"],
            "p_home_ml": p_home_ml,
            "p_away_ml": 1.0 - p_home_ml,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "p_home_runline": p_home_rl,
            "p_away_runline": p_away_rl,
            "exp_home_runs": lh,
            "exp_away_runs": la,
            "exp_total": lh + la,
        }
    )
