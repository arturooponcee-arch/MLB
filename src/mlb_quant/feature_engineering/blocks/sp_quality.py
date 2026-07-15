"""Bloque de calidad del abridor (temporada anterior).

Complementa a ``pitcher_form`` (forma reciente): esto mide el NIVEL del
pitcher — xwOBA en contra, xERA, whiff% — usando la temporada previa
completa (anti-leakage). Novatos quedan en null.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

_SQL = """
SELECT p.game_pk, p.team_id, p.outs_recorded,
       s.xwoba AS spq_xwoba_against,
       s.k_percent AS spq_k_pct,
       s.bb_percent AS spq_bb_pct,
       s.barrel_batted_rate AS spq_barrel_pct,
       s.whiff_percent AS spq_whiff_pct,
       e.xera AS spq_xera
FROM pitching_lines p
JOIN games g USING (game_pk)
LEFT JOIN savant_statcast_metrics s
    ON s.player_id = p.player_id
    AND s.year = g.season - 1
    AND s.player_type = 'pitcher'
    AND s.pa >= 100
LEFT JOIN savant_expected_stats e
    ON e.player_id = p.player_id
    AND e.year = g.season - 1
    AND e.player_type = 'pitcher'
    AND e.pa >= 100
WHERE p.is_starter
"""


@register
class SpQualityBlock(FeatureBlock):
    """Nivel del abridor según su temporada anterior completa.

    Salida por ``(game_pk, team_id)``: xwOBA en contra, xERA, K%, BB%,
    barrel% y whiff% del abridor (mínimo 100 bateadores enfrentados la
    temporada previa).
    """

    name: ClassVar[str] = "sp_quality"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = (
        "pitching_lines",
        "games",
        "savant_statcast_metrics",
        "savant_expected_stats",
    )

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula la calidad del abridor para cada apertura."""
        df = db.query(_SQL)
        # Juegos suspendidos pueden traer dos abridores: conservar el de
        # más outs (mismo criterio que pitcher_form).
        return (
            df.sort("outs_recorded", descending=True)
            .unique(subset=["game_pk", "team_id"], keep="first", maintain_order=True)
            .drop("outs_recorded")
        )
