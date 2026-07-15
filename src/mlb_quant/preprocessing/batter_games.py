"""Agregación de Statcast a línea por bateador-juego.

Base de los props de bateador: hits, bases totales, HR, walks y PA de
cada bateador en cada juego, derivados del pitch-by-pitch (la columna
``events`` solo trae valor en el último lanzamiento de cada turno).
"""

import logging

from mlb_quant.db import Database

logger = logging.getLogger(__name__)

_AGGREGATE_SQL = """
SELECT s.game_pk,
       s.batter AS player_id,
       CAST(s.game_date AS DATE) AS game_date,
       COUNT(s.events) AS pa,
       SUM(CASE WHEN s.events IN ('single','double','triple','home_run')
           THEN 1 ELSE 0 END)::INTEGER AS hits,
       SUM(CASE s.events WHEN 'single' THEN 1 WHEN 'double' THEN 2
           WHEN 'triple' THEN 3 WHEN 'home_run' THEN 4 ELSE 0 END)::INTEGER AS total_bases,
       SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END)::INTEGER AS home_runs,
       SUM(CASE WHEN s.events IN ('walk','intent_walk') THEN 1 ELSE 0 END)::INTEGER AS walks,
       SUM(CASE WHEN s.events = 'strikeout' THEN 1 ELSE 0 END)::INTEGER AS strikeouts
FROM statcast_pitches s
WHERE s.events IS NOT NULL
GROUP BY s.game_pk, s.batter, CAST(s.game_date AS DATE)
"""


def build_batter_games(db: Database) -> int:
    """Reconstruye la tabla ``batter_games`` desde ``statcast_pitches``.

    Args:
        db: Warehouse con Statcast ingerido.

    Returns:
        Filas escritas.

    Raises:
        ValueError: Si no existe ``statcast_pitches``.
    """
    if not db.table_exists("statcast_pitches"):
        msg = "Falta statcast_pitches. Corre antes: mlb ingest statcast"
        raise ValueError(msg)
    df = db.query(_AGGREGATE_SQL)
    written = db.upsert("batter_games", df, keys=["game_pk", "player_id"])
    logger.info("batter_games: %d filas.", written)
    return written
