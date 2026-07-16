"""Bloque Statcast del abridor: proceso, no resultado.

Métricas rodantes por apertura desde el pitch-by-pitch: whiff% (swings
fallados), CSW% (called strikes + whiffs sobre pitches), velocidad media
de la recta (FF+SI) con su tendencia, y xwOBA en contra. Complementa a
``sp_quality`` (temporada previa completa, lento para capturar forma) y
a ``pitcher_form`` (líneas de resultado: K, BB, ER).

xwOBA en contra: híbrido estándar — xwOBA estimado por speed/angle en
contacto y wOBA real en K/BB/HBP (``COALESCE`` sobre filas con
``woba_denom > 0``, que marcan exactamente el pitch final de cada PA).

Ratios desde sumas rolling laggeadas (idiom de ``pitcher_form``): la
apertura actual nunca entra a su propia ventana.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register
from mlb_quant.feature_engineering.rolling import RollingSpec, add_lagged_rolling

#: Ventanas (en aperturas) para las métricas Statcast.
STATCAST_WINDOWS: tuple[int, ...] = (3, 5, 10)

#: Descripciones de swing fallado (whiff).
WHIFF_DESCRIPTIONS: tuple[str, ...] = (
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
)

#: Descripciones que cuentan como swing (whiffs incluidos).
SWING_DESCRIPTIONS: tuple[str, ...] = (
    *WHIFF_DESCRIPTIONS,
    "foul",
    "foul_bunt",
    "hit_into_play",
    "missed_bunt",
    "bunt_foul_tip",
    "swinging_pitchout",
)

#: Columnas de conteo por apertura cuyas sumas rolling alimentan los ratios.
STATCAST_SUM_COLUMNS: tuple[str, ...] = (
    "pitches",
    "whiffs",
    "swings",
    "called_strikes",
    "fb_velo_sum",
    "fb_count",
    "xwoba_num",
    "xwoba_den",
)


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


#: Contadores crudos por (pitcher, juego). game_date de statcast es
#: TIMESTAMP: cast a DATE obligatorio para ordenar/juntar con games.
PER_START_SQL = f"""
SELECT s.pitcher AS player_id, s.game_pk, CAST(s.game_date AS DATE) AS game_date,
       COUNT(*) AS pitches,
       CAST(SUM(CASE WHEN s.description IN ({_sql_list(WHIFF_DESCRIPTIONS)})
                     THEN 1 ELSE 0 END) AS BIGINT) AS whiffs,
       CAST(SUM(CASE WHEN s.description IN ({_sql_list(SWING_DESCRIPTIONS)})
                     THEN 1 ELSE 0 END) AS BIGINT) AS swings,
       CAST(SUM(CASE WHEN s.description = 'called_strike' THEN 1 ELSE 0 END)
            AS BIGINT) AS called_strikes,
       CAST(SUM(CASE WHEN s.pitch_type IN ('FF', 'SI') THEN s.release_speed END)
            AS DOUBLE) AS fb_velo_sum,
       COUNT(CASE WHEN s.pitch_type IN ('FF', 'SI') THEN s.release_speed END) AS fb_count,
       CAST(SUM(CASE WHEN s.woba_denom > 0
                     THEN COALESCE(s.estimated_woba_using_speedangle, s.woba_value)
                END) AS DOUBLE) AS xwoba_num,
       CAST(SUM(CASE WHEN s.woba_denom > 0 THEN s.woba_denom END) AS BIGINT) AS xwoba_den
FROM statcast_pitches s
JOIN games g ON g.game_pk = s.game_pk AND g.game_type = 'R' AND g.status = 'Final'
GROUP BY 1, 2, 3
"""

#: Aperturas (con fecha) a las que se anclan las ventanas rolling.
STARTS_SQL = """
SELECT p.game_pk, p.team_id, p.player_id, p.outs_recorded, g.game_date
FROM pitching_lines p
JOIN games g USING (game_pk)
WHERE p.is_starter AND g.game_type = 'R' AND g.status = 'Final'
"""


def statcast_derived_expressions(
    windows: tuple[int, ...] = STATCAST_WINDOWS,
) -> list[pl.Expr]:
    """Ratios Statcast desde sumas rolling ``_{col}_sum_{w}``.

    Compartidas entre el bloque histórico y el camino de juegos futuros.
    Denominador 0 -> null (p. ej. ventana sin rectas lanzadas).
    """
    expressions = []
    for w in windows:
        expressions += [
            _ratio(f"_whiffs_sum_{w}", f"_swings_sum_{w}").alias(f"sp_whiff_pct_{w}"),
            (
                (pl.col(f"_called_strikes_sum_{w}") + pl.col(f"_whiffs_sum_{w}"))
                / _positive(f"_pitches_sum_{w}")
            ).alias(f"sp_csw_pct_{w}"),
            _ratio(f"_fb_velo_sum_sum_{w}", f"_fb_count_sum_{w}").alias(f"sp_fb_velo_{w}"),
            _ratio(f"_xwoba_num_sum_{w}", f"_xwoba_den_sum_{w}").alias(
                f"sp_xwoba_against_{w}"
            ),
        ]
    # Directo desde las sumas: las columnas sp_fb_velo_{w} se crean en el
    # mismo with_columns y aún no existen como columnas referenciables.
    expressions.append(
        (
            _ratio("_fb_velo_sum_sum_3", "_fb_count_sum_3")
            - _ratio("_fb_velo_sum_sum_10", "_fb_count_sum_10")
        ).alias("sp_fb_velo_trend")
    )
    return expressions


def attach_start_counts(db: Database) -> pl.DataFrame:
    """Aperturas con sus contadores Statcast (0 si el juego no tiene pitches).

    Left join: la fila de la apertura existe aunque falte el pitch-by-pitch
    del propio juego (las features salen de la HISTORIA previa). Contadores
    en 0 equivalen a excluir esa apertura de las sumas sin sacarla de la
    ventana. Compartida entre el bloque histórico y upcoming.
    """
    per_start = db.query(PER_START_SQL).drop("game_date")
    starts = db.query(STARTS_SQL).join(
        per_start, on=["game_pk", "player_id"], how="left"
    )
    return starts.with_columns(
        [pl.col(c).fill_null(0) for c in STATCAST_SUM_COLUMNS]
    )


def _positive(column: str) -> pl.Expr:
    return pl.when(pl.col(column) > 0).then(pl.col(column))


def _ratio(numerator: str, denominator: str) -> pl.Expr:
    return pl.col(numerator) / _positive(denominator)


@register
class SpStatcastBlock(FeatureBlock):
    """Métricas Statcast rodantes del abridor antes de cada apertura.

    Salida por ``(game_pk, team_id)``: ``sp_whiff_pct_{w}``,
    ``sp_csw_pct_{w}``, ``sp_fb_velo_{w}``, ``sp_xwoba_against_{w}``
    para cada ventana, más ``sp_fb_velo_trend`` (velo_3 - velo_10).
    """

    name: ClassVar[str] = "sp_statcast"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("statcast_pitches", "pitching_lines", "games")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula las métricas de proceso de cada abridor por apertura."""
        starts = attach_start_counts(db)
        specs = [
            RollingSpec(column, w, "sum", f"_{column}_sum_{w}")
            for w in STATCAST_WINDOWS
            for column in STATCAST_SUM_COLUMNS
        ]
        starts = add_lagged_rolling(
            starts, group_by="player_id", sort_by=["game_date", "game_pk"], specs=specs
        ).with_columns(statcast_derived_expressions())

        # Juegos suspendidos/reanudados pueden traer dos "abridores" en el
        # boxscore; conservar el de más outs (el abridor real).
        starts = starts.sort("outs_recorded", descending=True).unique(
            subset=["game_pk", "team_id"], keep="first", maintain_order=True
        )
        feature_columns = [c for c in starts.columns if c.startswith("sp_")]
        return starts.select(["game_pk", "team_id", *feature_columns])
