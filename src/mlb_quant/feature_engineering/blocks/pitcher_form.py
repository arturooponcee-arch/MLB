"""Bloque de forma del pitcher abridor: rendimiento, carga y descanso."""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register
from mlb_quant.feature_engineering.rolling import (
    RollingSpec,
    add_days_since_previous,
    add_lagged_rolling,
)

#: Constante FIP de liga (aprox. estable 3.0-3.2; se refinará por temporada).
FIP_CONSTANT = 3.15

#: Ventanas (en aperturas) para la forma del abridor.
WINDOWS: tuple[int, ...] = (3, 5, 10)

_APPEARANCES_SQL = """
SELECT p.player_id, p.game_pk, p.team_id, g.game_date, p.is_starter,
       p.outs_recorded, p.pitches_thrown, p.batters_faced,
       p.strike_outs, p.base_on_balls, p.home_runs, p.earned_runs
FROM pitching_lines p
JOIN games g USING (game_pk)
WHERE g.game_type = 'R' AND g.status = 'Final'
"""

#: Columnas cuyas sumas rolling alimentan las fórmulas derivadas.
SUM_COLUMNS: tuple[str, ...] = (
    "outs_recorded",
    "batters_faced",
    "strike_outs",
    "base_on_balls",
    "home_runs",
    "earned_runs",
)


def derived_expressions(windows: tuple[int, ...] = WINDOWS) -> list[pl.Expr]:
    """Fórmulas de K%, BB%, IP, FIP y ERA desde sumas rolling por ventana.

    Espera columnas ``_{col}_sum_{w}`` y ``_outs_mean_{w}`` ya calculadas.
    Compartidas entre el bloque histórico y el builder de juegos futuros.
    """
    expressions = []
    for w in windows:
        ip = pl.col(f"_outs_recorded_sum_{w}") / 3.0
        bf = pl.col(f"_batters_faced_sum_{w}")
        expressions += [
            (pl.col(f"_strike_outs_sum_{w}") / bf).alias(f"sp_k_pct_{w}"),
            (pl.col(f"_base_on_balls_sum_{w}") / bf).alias(f"sp_bb_pct_{w}"),
            (pl.col(f"_outs_mean_{w}") / 3.0).alias(f"sp_ip_per_start_{w}"),
            (
                (
                    13 * pl.col(f"_home_runs_sum_{w}")
                    + 3 * pl.col(f"_base_on_balls_sum_{w}")
                    - 2 * pl.col(f"_strike_outs_sum_{w}")
                )
                / ip
                + FIP_CONSTANT
            ).alias(f"sp_fip_{w}"),
            (9 * pl.col(f"_earned_runs_sum_{w}") / ip).alias(f"sp_era_{w}"),
        ]
    return expressions


@register
class PitcherFormBlock(FeatureBlock):
    """Features del abridor antes de cada apertura.

    Salida por ``(game_pk, team_id)`` (una fila por abridor real del
    equipo): descanso en días, K%/BB% rolling, FIP y ERA rolling,
    lanzamientos e innings promedio recientes.
    """

    name: ClassVar[str] = "pitcher_form"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("pitching_lines", "games")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula la forma de cada abridor antes de cada apertura."""
        appearances = db.query(_APPEARANCES_SQL)

        # Descanso: días desde la apariencia anterior (incluye relevos).
        appearances = add_days_since_previous(
            appearances,
            group_by="player_id",
            date_column="game_date",
            alias="sp_days_rest",
        )

        # Rolling sobre APERTURAS solamente: ratios desde sumas acumuladas
        # (más estable que promediar ratios por juego).
        starts = appearances.filter(pl.col("is_starter"))
        specs = [
            RollingSpec(column, w, "sum", f"_{column}_sum_{w}")
            for w in WINDOWS
            for column in SUM_COLUMNS
        ] + [
            spec
            for w in WINDOWS
            for spec in (
                RollingSpec("pitches_thrown", w, "mean", f"sp_pitches_avg_{w}"),
                RollingSpec("outs_recorded", w, "mean", f"_outs_mean_{w}"),
            )
        ]
        starts = add_lagged_rolling(
            starts, group_by="player_id", sort_by=["game_date", "game_pk"], specs=specs
        )
        starts = starts.with_columns(derived_expressions())

        # Juegos suspendidos/reanudados pueden traer dos "abridores" en el
        # boxscore; conservar el de más outs (el abridor real).
        starts = starts.sort("outs_recorded", descending=True).unique(
            subset=["game_pk", "team_id"], keep="first", maintain_order=True
        )

        feature_columns = ["sp_days_rest"] + [c for c in starts.columns if c.startswith("sp_")]
        return starts.select(["game_pk", "team_id", *dict.fromkeys(feature_columns)])
