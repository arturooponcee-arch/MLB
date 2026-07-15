"""Ensamblado de la matriz de features por juego.

Une la salida de todos los bloques registrados sobre la base de juegos:
bloques de grano ``game`` se unen 1:1 por ``game_pk``; los de grano
``team_game`` se unen dos veces (equipo local y visitante) con prefijos
``home_`` / ``away_``.
"""

import logging
from datetime import date

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, get_blocks

logger = logging.getLogger(__name__)

_BASE_SQL = """
SELECT game_pk, game_date, season, home_team_id, away_team_id,
       home_score, away_score
FROM games
WHERE status = 'Final' AND game_type = 'R'
  AND game_date BETWEEN DATE '{start}' AND DATE '{end}'
"""


def build_game_features(
    db: Database,
    start_date: date,
    end_date: date,
    block_names: list[str] | None = None,
) -> pl.DataFrame:
    """Construye la matriz de features para los juegos de un rango.

    Los bloques calculan sobre TODO el warehouse (necesitan historia
    anterior al rango) y aquí se filtra a los juegos objetivo.

    Args:
        db: Base de datos con las tablas de ingesta.
        start_date: Primer día del rango (inclusive).
        end_date: Último día del rango (inclusive).
        block_names: Bloques a usar; ``None`` = todos los registrados.

    Returns:
        Una fila por juego: claves, labels (``home_score``, ``away_score``)
        y todas las features. Los juegos sin historia quedan con nulls.

    Raises:
        ValueError: Si falta alguna tabla requerida por un bloque.
    """
    blocks = get_blocks(block_names)
    missing = {b.name: b.check_requirements(db) for b in blocks}
    problems = {name: tables for name, tables in missing.items() if tables}
    if problems:
        msg = f"Faltan tablas en el warehouse: {problems}. Corre mlb ingest."
        raise ValueError(msg)

    base = db.query(_BASE_SQL.format(start=start_date, end=end_date))
    logger.info("Base: %d juegos entre %s y %s.", len(base), start_date, end_date)

    for block in blocks:
        features = block.build(db)
        base = _join_block(base, block, features)
        logger.info("Bloque %s: %d columnas acumuladas.", block.name, len(base.columns))
    return base


def _join_block(
    base: pl.DataFrame, block: FeatureBlock, features: pl.DataFrame
) -> pl.DataFrame:
    """Une la salida de un bloque a la base según su grano."""
    if block.grain == "game":
        return base.join(features, on="game_pk", how="left")

    feature_cols = [c for c in features.columns if c not in ("game_pk", "team_id")]
    for side in ("home", "away"):
        renamed = features.rename(
            {c: f"{side}_{c}" for c in feature_cols} | {"team_id": f"{side}_team_id"}
        )
        base = base.join(renamed, on=["game_pk", f"{side}_team_id"], how="left")
    return base
