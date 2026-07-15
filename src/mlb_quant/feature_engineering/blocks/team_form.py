"""Bloque de forma del equipo: resultados y carreras recientes."""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register
from mlb_quant.feature_engineering.rolling import RollingSpec, add_lagged_rolling

#: Ventanas (en juegos) para la forma del equipo.
WINDOWS: tuple[int, ...] = (5, 10, 30)

_LONG_SQL = """
SELECT game_pk, game_date, home_team_id AS team_id,
       home_score AS runs_scored, away_score AS runs_allowed,
       CAST(home_score > away_score AS INTEGER) AS win
FROM games
WHERE status = 'Final' AND game_type = 'R'
UNION ALL
SELECT game_pk, game_date, away_team_id AS team_id,
       away_score AS runs_scored, home_score AS runs_allowed,
       CAST(away_score > home_score AS INTEGER) AS win
FROM games
WHERE status = 'Final' AND game_type = 'R'
"""


@register
class TeamFormBlock(FeatureBlock):
    """Win rate y carreras anotadas/permitidas en ventanas recientes.

    Salida por ``(game_pk, team_id)``: ``team_win_rate_{w}``,
    ``team_runs_scored_{w}``, ``team_runs_allowed_{w}`` para cada ventana.
    """

    name: ClassVar[str] = "team_form"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("games",)

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula la forma de cada equipo antes de cada juego."""
        long = db.query(_LONG_SQL)
        specs = [
            spec
            for w in WINDOWS
            for spec in (
                RollingSpec("win", w, "mean", f"team_win_rate_{w}"),
                RollingSpec("runs_scored", w, "mean", f"team_runs_scored_{w}"),
                RollingSpec("runs_allowed", w, "mean", f"team_runs_allowed_{w}"),
            )
        ]
        result = add_lagged_rolling(
            long, group_by="team_id", sort_by=["game_date", "game_pk"], specs=specs
        )
        return result.drop("game_date", "runs_scored", "runs_allowed", "win")
