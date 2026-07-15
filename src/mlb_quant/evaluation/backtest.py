"""Backtest walk-forward: la única validación honesta para apuestas.

Reentrena cada mes usando SOLO juegos anteriores al mes y predice los
juegos de ese mes. Nunca hay información futura en el entrenamiento.
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from mlb_quant.evaluation.metrics import probability_metrics
from mlb_quant.models.ensemble import CalibratedEnsembleWinModel
from mlb_quant.models.logistic import LogisticWinModel
from mlb_quant.models.markets import market_probabilities
from mlb_quant.models.poisson import PoissonRunsModel

logger = logging.getLogger(__name__)

#: Mínimo de juegos de entrenamiento antes de emitir predicciones.
MIN_TRAIN_GAMES = 500


@dataclass(frozen=True)
class BacktestResult:
    """Resultado de un backtest walk-forward.

    Attributes:
        predictions: Una fila por juego predicho, con probabilidades,
            lambdas y resultados observados.
        metrics: Métricas agregadas por modelo/mercado.
    """

    predictions: pl.DataFrame
    metrics: dict[str, dict[str, float]]


def _month_starts(start: date, end: date) -> list[date]:
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append(current)
        current = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
    return months


def walk_forward(
    features: pl.DataFrame,
    start: date,
    end: date,
    total_line: float = 8.5,
    min_train_games: int = MIN_TRAIN_GAMES,
) -> BacktestResult:
    """Corre el backtest mensual sobre una matriz de features.

    Args:
        features: Matriz completa (el entrenamiento usa historia anterior
            al rango evaluado; el rango define qué meses se predicen).
        start: Primer día a predecir.
        end: Último día a predecir.
        total_line: Línea de over/under a evaluar.
        min_train_games: Historia mínima para emitir predicciones.

    Returns:
        Predicciones por juego y métricas agregadas.

    Raises:
        ValueError: Si ningún mes alcanza la historia mínima.
    """
    frames: list[pl.DataFrame] = []
    for month_start in _month_starts(start, end):
        train = features.filter(pl.col("game_date") < month_start)
        month_end = _month_starts(month_start, date(month_start.year + 1, 1, 1))[1]
        test = features.filter(
            (pl.col("game_date") >= max(month_start, start))
            & (pl.col("game_date") < month_end)
            & (pl.col("game_date") <= end)
        )
        if len(train) < min_train_games or test.is_empty():
            logger.info(
                "Mes %s omitido (train=%d, test=%d).", month_start, len(train), len(test)
            )
            continue

        poisson = PoissonRunsModel().fit(train)
        logistic = LogisticWinModel().fit(train)
        ensemble = CalibratedEnsembleWinModel().fit(train)

        lambdas = poisson.predict_lambdas(test)
        markets = market_probabilities(lambdas, total_line=total_line)
        logit = logistic.predict_home_win(test).rename({"p_home_win": "p_home_logistic"})
        ens = ensemble.predict_home_win(test).rename({"p_home_win": "p_home_ensemble"})

        frames.append(
            test.select("game_pk", "game_date", "home_score", "away_score")
            .join(markets, on="game_pk")
            .join(logit, on="game_pk")
            .join(ens, on="game_pk")
        )
        logger.info(
            "Mes %s: train=%d juegos, predichos=%d.", month_start, len(train), len(test)
        )

    if not frames:
        msg = "Ningún mes con historia suficiente para el backtest."
        raise ValueError(msg)

    predictions = pl.concat(frames)
    return BacktestResult(predictions=predictions, metrics=_evaluate(predictions, total_line))


def _evaluate(
    predictions: pl.DataFrame, total_line: float
) -> dict[str, dict[str, float]]:
    """Métricas de moneyline (ambos modelos) y over/under."""
    home_win = (
        (predictions["home_score"] > predictions["away_score"]).to_numpy().astype(int)
    )
    over = (
        ((predictions["home_score"] + predictions["away_score"]) > total_line)
        .to_numpy()
        .astype(int)
    )
    total_mae = float(
        np.abs(
            (predictions["home_score"] + predictions["away_score"]).to_numpy()
            - predictions["exp_total"].to_numpy()
        ).mean()
    )
    metrics = {
        "moneyline_poisson": probability_metrics(
            home_win, predictions["p_home_ml"].to_numpy()
        ),
        "moneyline_logistic": probability_metrics(
            home_win, predictions["p_home_logistic"].to_numpy()
        ),
        "moneyline_ensemble": probability_metrics(
            home_win, predictions["p_home_ensemble"].to_numpy()
        ),
        f"over_{total_line}_poisson": probability_metrics(
            over, predictions["p_over"].to_numpy()
        ),
    }
    metrics["totales"] = {"mae_total_runs": total_mae, "n": float(len(predictions))}
    return metrics
