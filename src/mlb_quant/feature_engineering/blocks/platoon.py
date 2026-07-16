"""Bloque platoon: ofensiva del equipo contra la mano del abridor rival.

Para cada ``(game_pk, equipo bateador)``: xwOBA del equipo (todas las PA
de sus bateadores) contra lanzadores de la misma mano que el abridor
RIVAL, en la ventana estricta de los 365 días previos, más el conteo de
PA como feature de tamaño de muestra.

Diseño a nivel de equipo (no de lineup): funciona idéntico para juegos
futuros donde el lineup no se conoce, y los bateadores ambidiestros se
manejan solos porque el split usa el ``stand`` real de cada PA.

Anti-fuga: la ventana ``[fecha-365d, fecha)`` excluye el propio día —
también cubre doubleheaders. La mano del abridor es la moda de
``p_throws`` en toda su carrera Statcast (lahman no cruza con ids MLBAM).
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

#: Días de la ventana retrospectiva de producción por mano.
LOOKBACK_DAYS = 365

#: Mano dominante por pitcher (moda sobre toda su carrera Statcast).
HANDS_SQL = """
SELECT pitcher AS player_id, mode(p_throws) AS hand
FROM statcast_pitches
GROUP BY 1
"""

#: Producción ofensiva diaria por (equipo bateador, mano del pitcher).
#: woba_denom > 0 marca el pitch final de cada PA; el equipo bateador se
#: deriva de la mitad del inning (Bot = batea el local).
TEAM_DAILY_SQL = """
SELECT CASE WHEN s.inning_topbot = 'Bot' THEN g.home_team_id
            ELSE g.away_team_id END AS team_id,
       s.p_throws AS hand,
       CAST(s.game_date AS DATE) AS game_date,
       CAST(SUM(COALESCE(s.estimated_woba_using_speedangle, s.woba_value))
            AS DOUBLE) AS num,
       CAST(SUM(s.woba_denom) AS BIGINT) AS den,
       COUNT(*) AS pa
FROM statcast_pitches s
JOIN games g ON g.game_pk = s.game_pk AND g.game_type = 'R' AND g.status = 'Final'
WHERE s.woba_denom > 0
GROUP BY 1, 2, 3
"""

_PLATOON_SQL = f"""
WITH hands AS ({HANDS_SQL}),
team_daily AS ({TEAM_DAILY_SQL}),
starters AS (
    SELECT p.game_pk, p.team_id, p.player_id,
           g.game_date, g.home_team_id, g.away_team_id,
           ROW_NUMBER() OVER (
               PARTITION BY p.game_pk, p.team_id ORDER BY p.outs_recorded DESC
           ) AS rn
    FROM pitching_lines p
    JOIN games g USING (game_pk)
    WHERE p.is_starter AND g.game_type = 'R' AND g.status = 'Final'
),
opposing AS (
    -- Cada equipo BATEADOR con la mano del abridor rival.
    SELECT s.game_pk,
           CASE WHEN s.team_id = s.home_team_id THEN s.away_team_id
                ELSE s.home_team_id END AS team_id,
           s.game_date,
           h.hand AS opp_hand
    FROM starters s
    JOIN hands h ON h.player_id = s.player_id
    WHERE s.rn = 1
)
SELECT o.game_pk, o.team_id,
       SUM(t.num) / NULLIF(SUM(t.den), 0) AS lineup_xwoba_vs_hand,
       COALESCE(SUM(t.pa), 0) AS lineup_pa_vs_hand
FROM opposing o
LEFT JOIN team_daily t
    ON t.team_id = o.team_id AND t.hand = o.opp_hand
   AND t.game_date >= o.game_date - INTERVAL {LOOKBACK_DAYS} DAY
   AND t.game_date < o.game_date
GROUP BY 1, 2
"""


@register
class PlatoonBlock(FeatureBlock):
    """Producción del equipo contra la mano del abridor rival.

    Salida por ``(game_pk, team_id)``: ``lineup_xwoba_vs_hand`` (null sin
    historia) y ``lineup_pa_vs_hand`` (tamaño de muestra, 0 sin historia).
    Juegos cuyo abridor rival no tiene mano conocida quedan fuera (el
    builder los deja en null vía left join).
    """

    name: ClassVar[str] = "platoon"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("statcast_pitches", "pitching_lines", "games")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula el split por mano de cada equipo antes de cada juego."""
        return db.query(_PLATOON_SQL)
