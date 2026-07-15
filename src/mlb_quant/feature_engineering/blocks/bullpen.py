"""Bloque de fatiga del bullpen.

Lanzamientos de los relevistas del equipo en los 1 y 3 días anteriores
al juego. Bullpen cargado = relevistas cansados o indisponibles.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

_SQL = """
WITH team_games AS (
    SELECT game_pk, game_date, home_team_id AS team_id
    FROM games WHERE game_type = 'R'
    UNION ALL
    SELECT game_pk, game_date, away_team_id AS team_id
    FROM games WHERE game_type = 'R'
),
relief AS (
    SELECT p.team_id, g.game_date,
           SUM(p.pitches_thrown) AS relief_pitches,
           COUNT(*) AS relievers_used
    FROM pitching_lines p
    JOIN games g USING (game_pk)
    WHERE NOT p.is_starter AND g.game_type = 'R' AND g.status = 'Final'
    GROUP BY p.team_id, g.game_date
)
SELECT tg.game_pk, tg.team_id,
       COALESCE(SUM(CASE WHEN r.game_date = tg.game_date - 1
                         THEN r.relief_pitches END), 0) AS bullpen_pitches_1d,
       COALESCE(SUM(r.relief_pitches), 0) AS bullpen_pitches_3d,
       COALESCE(SUM(CASE WHEN r.game_date = tg.game_date - 1
                         THEN r.relievers_used END), 0) AS bullpen_relievers_1d
FROM team_games tg
LEFT JOIN relief r
    ON r.team_id = tg.team_id
    AND r.game_date BETWEEN tg.game_date - 3 AND tg.game_date - 1
GROUP BY tg.game_pk, tg.team_id
"""


@register
class BullpenFatigueBlock(FeatureBlock):
    """Carga reciente del bullpen antes de cada juego.

    Salida por ``(game_pk, team_id)``: lanzamientos de relevo del día
    anterior y de los últimos 3 días, y relevistas usados ayer. Solo mira
    días ANTERIORES a la fecha del juego (anti-leakage).
    """

    name: ClassVar[str] = "bullpen_fatigue"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("pitching_lines", "games")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula la fatiga del bullpen para cada juego."""
        return db.query(_SQL)
