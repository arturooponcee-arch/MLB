"""Bloque de parque: física del estadio + run factor propio.

El run factor se calcula SOLO con temporadas anteriores a la del juego
(anti-leakage): carreras promedio en el estadio vs promedio de la liga.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

_PHYSICAL_SQL = """
SELECT g.game_pk, g.season, g.venue_id,
       v.elevation AS park_elevation,
       v.azimuth_angle AS park_azimuth,
       CAST(v.roof_type IN ('Dome', 'Retractable') AS INTEGER) AS park_has_roof,
       v.center AS park_center,
       v.left_line AS park_left_line,
       v.right_line AS park_right_line
FROM games g
LEFT JOIN venues v ON g.venue_id = v.venue_id AND v.season = g.season
"""

_VENUE_SEASON_RUNS_SQL = """
SELECT venue_id, season,
       AVG(home_score + away_score) AS venue_runs_pg,
       COUNT(*) AS games_played
FROM games
WHERE status = 'Final' AND game_type = 'R'
GROUP BY venue_id, season
"""


@register
class ParkBlock(FeatureBlock):
    """Características del estadio por juego.

    Salida por ``game_pk``: elevación, azimuth, techo, dimensiones y
    ``park_run_factor`` (histórico previo del estadio / liga; null si el
    estadio no tiene temporadas anteriores en el warehouse).
    """

    name: ClassVar[str] = "park"
    grain: ClassVar[Grain] = "game"
    requires: ClassVar[tuple[str, ...]] = ("games", "venues")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula features de parque para cada juego."""
        physical = db.query(_PHYSICAL_SQL)
        per_season = db.query(_VENUE_SEASON_RUNS_SQL)

        factors = self._run_factors(per_season).with_columns(
            pl.col("venue_id").cast(physical.schema["venue_id"]),
            pl.col("season").cast(physical.schema["season"]),
        )
        result = physical.join(factors, on=["venue_id", "season"], how="left")
        return result.drop("venue_id", "season")

    @staticmethod
    def _run_factors(per_season: pl.DataFrame) -> pl.DataFrame:
        """Run factor por (venue, season) usando solo temporadas previas."""
        weighted = (pl.col("venue_runs_pg") * pl.col("games_played")).sum()
        total = pl.col("games_played").sum()
        rows = []
        for season in sorted(per_season["season"].unique().to_list()):
            past = per_season.filter(pl.col("season") < season)
            if past.is_empty():
                continue
            league_avg = past.select(weighted / total).item()
            rows.append(
                past.group_by("venue_id")
                .agg((weighted / total).alias("venue_avg"))
                .with_columns(
                    pl.lit(season).alias("season"),
                    (pl.col("venue_avg") / league_avg).alias("park_run_factor"),
                )
                .select("venue_id", "season", "park_run_factor")
            )
        if not rows:
            return pl.DataFrame(
                schema={
                    "venue_id": pl.Int64,
                    "season": pl.Int64,
                    "park_run_factor": pl.Float64,
                }
            )
        return pl.concat(rows)
