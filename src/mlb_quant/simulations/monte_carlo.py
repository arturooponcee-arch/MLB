"""Simulación Monte Carlo de juegos desde tasas Poisson.

Ventajas sobre la malla analítica de :mod:`mlb_quant.models.markets`:
extra innings reales (los empates se juegan entrada a entrada), run line
correcto tras extras, primeras 5 entradas y cuantiles de distribución.

Aproximación F5: tasa escalada 5/9 (ignora que el abridor suele ser
mejor que el bullpen; sesgo pequeño y estable).
"""

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

#: Entradas extra máximas simuladas antes del desempate forzado.
MAX_EXTRA_INNINGS = 30


@dataclass(frozen=True)
class SimulationConfig:
    """Parámetros de la simulación.

    Attributes:
        n_sims: Simulaciones por juego.
        seed: Semilla del generador (reproducibilidad).
        total_line: Línea de over/under del juego completo.
        run_line: Hándicap del favorito local.
        team_total_line: Línea de carreras por equipo.
        f5_total_line: Línea de over/under de las primeras 5 entradas.
        chunk_size: Juegos por lote (control de memoria).
    """

    n_sims: int = 10_000
    seed: int = 42
    total_line: float = 8.5
    run_line: float = 1.5
    team_total_line: float = 4.5
    f5_total_line: float = 4.5
    chunk_size: int = 500


def simulate_games(
    lambdas: pl.DataFrame, config: SimulationConfig | None = None
) -> pl.DataFrame:
    """Simula cada juego ``n_sims`` veces y resume los mercados.

    Args:
        lambdas: ``game_pk``, ``lambda_home``, ``lambda_away``.
        config: Parámetros; por defecto :class:`SimulationConfig`.

    Returns:
        Una fila por juego con probabilidades de todos los mercados,
        esperanzas y cuantiles de la distribución de carreras.
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    chunks = []
    for offset in range(0, len(lambdas), cfg.chunk_size):
        chunk = lambdas.slice(offset, cfg.chunk_size)
        chunks.append(_simulate_chunk(chunk, cfg, rng))
    result = pl.concat(chunks)
    logger.info(
        "Simulados %d juegos x %d sims (semilla %d).", len(result), cfg.n_sims, cfg.seed
    )
    return result


def _simulate_chunk(
    lambdas: pl.DataFrame, cfg: SimulationConfig, rng: np.random.Generator
) -> pl.DataFrame:
    """Simula un lote de juegos de una pasada vectorizada."""
    lh = lambdas["lambda_home"].to_numpy()[:, None]  # (n, 1)
    la = lambdas["lambda_away"].to_numpy()[:, None]
    n = lh.shape[0]
    shape = (n, cfg.n_sims)

    home = rng.poisson(np.broadcast_to(lh, shape))
    away = rng.poisson(np.broadcast_to(la, shape))

    # Extra innings: los empates se juegan entrada a entrada (tasa /9).
    ties = home == away
    for _ in range(MAX_EXTRA_INNINGS):
        if not ties.any():
            break
        home = home + rng.poisson(np.broadcast_to(lh / 9.0, shape)) * ties
        away = away + rng.poisson(np.broadcast_to(la / 9.0, shape)) * ties
        ties = home == away
    if ties.any():  # desempate forzado (probabilidad ~0)
        favorite = rng.random(shape) < lh / (lh + la)
        home = home + (ties & favorite)
        away = away + (ties & ~favorite)

    # Primeras 5 entradas: tasa escalada, empate permitido.
    f5_home = rng.poisson(np.broadcast_to(lh * 5 / 9, shape))
    f5_away = rng.poisson(np.broadcast_to(la * 5 / 9, shape))

    total = home + away
    margin = home - away
    f5_total = f5_home + f5_away

    return pl.DataFrame(
        {
            "game_pk": lambdas["game_pk"],
            "sim_p_home_ml": (margin > 0).mean(axis=1),
            "sim_p_away_ml": (margin < 0).mean(axis=1),
            "sim_p_over": (total > cfg.total_line).mean(axis=1),
            "sim_p_under": (total < cfg.total_line).mean(axis=1),
            "sim_p_home_runline": (margin > cfg.run_line).mean(axis=1),
            "sim_p_away_runline": (margin < cfg.run_line).mean(axis=1),
            "sim_p_home_team_over": (home > cfg.team_total_line).mean(axis=1),
            "sim_p_away_team_over": (away > cfg.team_total_line).mean(axis=1),
            "sim_p_f5_home_ml": (f5_home > f5_away).mean(axis=1),
            "sim_p_f5_tie": (f5_home == f5_away).mean(axis=1),
            "sim_p_f5_over": (f5_total > cfg.f5_total_line).mean(axis=1),
            "sim_exp_total": total.mean(axis=1),
            "sim_exp_margin": margin.mean(axis=1),
            "sim_total_p10": np.percentile(total, 10, axis=1),
            "sim_total_p50": np.percentile(total, 50, axis=1),
            "sim_total_p90": np.percentile(total, 90, axis=1),
        }
    )
