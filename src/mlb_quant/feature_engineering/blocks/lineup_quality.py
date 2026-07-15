"""Bloque de calidad del lineup titular.

Promedia las métricas Statcast de los 9 titulares usando la TEMPORADA
ANTERIOR (anti-leakage: la agregada de la temporada en curso incluye
juegos futuros). Novatos sin temporada previa quedan fuera del promedio;
``lineup_known_batters`` mide la cobertura.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

_SQL = """
SELECT l.game_pk, l.team_id,
       AVG(s.xwoba) AS lineup_xwoba,
       AVG(s.woba) AS lineup_woba,
       AVG(s.k_percent) AS lineup_k_pct,
       AVG(s.bb_percent) AS lineup_bb_pct,
       AVG(s.barrel_batted_rate) AS lineup_barrel_pct,
       AVG(s.hard_hit_percent) AS lineup_hard_hit_pct,
       AVG(s.whiff_percent) AS lineup_whiff_pct,
       AVG(s.oz_swing_percent) AS lineup_chase_pct,
       COUNT(s.player_id) AS lineup_known_batters
FROM lineups l
JOIN games g USING (game_pk)
LEFT JOIN savant_statcast_metrics s
    ON s.player_id = l.player_id
    AND s.year = g.season - 1
    AND s.player_type = 'batter'
    AND s.pa >= 100
WHERE l.is_starting_lineup
GROUP BY l.game_pk, l.team_id
"""


@register
class LineupQualityBlock(FeatureBlock):
    """Calidad ofensiva del lineup titular (temporada anterior).

    Salida por ``(game_pk, team_id)``: xwOBA, K%, BB%, barrel%, hard
    hit%, whiff%, chase% promedio de los titulares con al menos 100 PA
    la temporada previa, más el número de titulares con datos.
    """

    name: ClassVar[str] = "lineup_quality"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("lineups", "games", "savant_statcast_metrics")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula la calidad del lineup para cada juego con alineación."""
        return db.query(_SQL)
