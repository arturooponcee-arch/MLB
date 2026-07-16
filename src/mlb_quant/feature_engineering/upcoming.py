"""Features para juegos FUTUROS (programados, sin jugar).

Misma semántica que el builder histórico — estado ANTES del juego — pero
"as of" hoy: la forma del equipo/abridor incluye todos sus juegos ya
completados. Los nombres de columna coinciden exactamente con los del
entrenamiento; lo que no se conoce antes del juego (lineup confirmado)
queda ausente y el modelo lo imputa.

Abridor: se usa el pitcher PROBABLE del calendario (en el histórico es
el abridor real; la discrepancia probable/real es ruido pequeño y
conocido del pronóstico pregame).
"""

import logging
from datetime import date

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock
from mlb_quant.feature_engineering.blocks.bullpen import BullpenFatigueBlock
from mlb_quant.feature_engineering.blocks.elo import ELO_GAMES_SQL, compute_elo
from mlb_quant.feature_engineering.blocks.park import ParkBlock
from mlb_quant.feature_engineering.blocks.pitcher_form import (
    SUM_COLUMNS,
    WINDOWS,
    derived_expressions,
)
from mlb_quant.feature_engineering.blocks.rest_travel import (
    VENUE_COORDS_SQL,
    haversine_km,
    team_travel_history,
    tz_offset,
)
from mlb_quant.feature_engineering.blocks.team_form import _LONG_SQL as TEAM_LONG_SQL
from mlb_quant.feature_engineering.blocks.team_form import WINDOWS as TEAM_WINDOWS
from mlb_quant.feature_engineering.blocks.weather import WeatherBlock
from mlb_quant.feature_engineering.builder import _join_block

logger = logging.getLogger(__name__)

_BASE_SQL = """
SELECT game_pk, game_date, game_datetime_utc, season,
       home_team_id, away_team_id, home_team_name, away_team_name,
       venue_id, venue_name, status,
       home_probable_pitcher_id, away_probable_pitcher_id,
       home_probable_pitcher_name, away_probable_pitcher_name
FROM games
WHERE game_type = 'R' AND status != 'Final'
  AND game_date BETWEEN DATE '{start}' AND DATE '{end}'
"""

_APPEARANCES_SQL = """
SELECT p.player_id, p.game_pk, g.game_date, p.is_starter,
       p.outs_recorded, p.pitches_thrown, p.batters_faced,
       p.strike_outs, p.base_on_balls, p.home_runs, p.earned_runs
FROM pitching_lines p
JOIN games g USING (game_pk)
WHERE g.game_type = 'R' AND g.status = 'Final'
"""

_SAVANT_PITCHER_SQL = """
SELECT s.player_id, s.year,
       s.xwoba AS spq_xwoba_against,
       s.k_percent AS spq_k_pct,
       s.bb_percent AS spq_bb_pct,
       s.barrel_batted_rate AS spq_barrel_pct,
       s.whiff_percent AS spq_whiff_pct,
       e.xera AS spq_xera
FROM savant_statcast_metrics s
LEFT JOIN savant_expected_stats e
    ON e.player_id = s.player_id AND e.year = s.year AND e.player_type = 'pitcher'
WHERE s.player_type = 'pitcher' AND s.pa >= 100
"""


def build_upcoming_features(db: Database, start: date, end: date) -> pl.DataFrame:
    """Construye la matriz de features de los juegos programados de un rango.

    Args:
        db: Warehouse con calendario ingerido para el rango (incluida el
            clima forecast si se quiere el bloque de clima completo).
        start: Primer día (inclusive).
        end: Último día (inclusive).

    Returns:
        Una fila por juego programado: metadatos para mostrar (equipos,
        probables, hora) + features con los mismos nombres del
        entrenamiento.
    """
    base = db.query(_BASE_SQL.format(start=start, end=end))
    if base.is_empty():
        return base
    logger.info("Juegos programados %s -> %s: %d.", start, end, len(base))

    # Bloques cuyo SQL ya cubre juegos no finalizados: reutilizados tal cual.
    for block in (ParkBlock(), WeatherBlock(), BullpenFatigueBlock()):
        base = _join_reusable(base, block, db)

    base = _join_team_form(base, db)
    base = _join_pitchers(base, db)
    base = _join_elo(base, db)
    base = _join_rest_travel(base, db)
    return base


def _join_reusable(base: pl.DataFrame, block: FeatureBlock, db: Database) -> pl.DataFrame:
    missing = block.check_requirements(db)
    if missing:
        logger.warning("Bloque %s omitido: faltan tablas %s.", block.name, missing)
        return base
    return _join_block(base, block, block.build(db))


def _join_team_form(base: pl.DataFrame, db: Database) -> pl.DataFrame:
    """Forma actual de cada equipo (últimos N juegos completados)."""
    long = db.query(TEAM_LONG_SQL).sort("game_date", "game_pk")
    aggregations = []
    for w in TEAM_WINDOWS:
        aggregations += [
            pl.col("win").tail(w).mean().alias(f"team_win_rate_{w}"),
            pl.col("runs_scored").tail(w).mean().alias(f"team_runs_scored_{w}"),
            pl.col("runs_allowed").tail(w).mean().alias(f"team_runs_allowed_{w}"),
        ]
    state = long.group_by("team_id").agg(aggregations)
    feature_cols = [c for c in state.columns if c != "team_id"]
    for side in ("home", "away"):
        renamed = state.rename(
            {c: f"{side}_{c}" for c in feature_cols} | {"team_id": f"{side}_team_id"}
        )
        base = base.join(renamed, on=f"{side}_team_id", how="left")
    return base


def _join_elo(base: pl.DataFrame, db: Database) -> pl.DataFrame:
    """Rating Elo vigente de cada equipo (tras su último juego completado)."""
    _, ratings = compute_elo(db.query(ELO_GAMES_SQL))
    state = pl.DataFrame(
        {"team_id": list(ratings.keys()), "elo_pre": list(ratings.values())},
        schema={"team_id": pl.Int64, "elo_pre": pl.Float64},
    )
    for side in ("home", "away"):
        base = base.join(
            state.rename({"team_id": f"{side}_team_id", "elo_pre": f"{side}_elo_pre"}),
            on=f"{side}_team_id",
            how="left",
        )
    return base


def _join_rest_travel(base: pl.DataFrame, db: Database) -> pl.DataFrame:
    """Descanso/viaje de cada equipo desde su último juego completado."""
    if not db.table_exists("venues"):
        logger.warning("Bloque rest_travel omitido: falta la tabla venues.")
        return base
    history = team_travel_history(db)
    # Estado por equipo = su último juego completado (historial ya ordenado).
    state = history.group_by("team_id", maintain_order=True).agg(
        pl.col("game_date").last().alias("_last_date"),
        pl.col("latitude").last().alias("_last_lat"),
        pl.col("longitude").last().alias("_last_lon"),
        pl.col("is_away").last().alias("_last_away"),
        pl.col("road_trip_len").last().alias("_last_trip"),
    )
    coords = db.query(VENUE_COORDS_SQL)
    base = base.join(coords, on="venue_id", how="left")
    for side in ("home", "away"):
        side_state = state.rename(
            {c: f"{side}{c}" for c in state.columns if c != "team_id"}
        )
        base = base.join(
            side_state, left_on=f"{side}_team_id", right_on="team_id", how="left"
        )
        is_away = 1 if side == "away" else 0
        base = base.with_columns(
            (pl.col("game_date") - pl.col(f"{side}_last_date"))
            .dt.total_days()
            .alias(f"{side}_team_days_rest"),
            haversine_km(
                pl.col(f"{side}_last_lat"),
                pl.col(f"{side}_last_lon"),
                pl.col("latitude"),
                pl.col("longitude"),
            ).alias(f"{side}_travel_km"),
            (tz_offset(pl.col("longitude")) - tz_offset(pl.col(f"{side}_last_lon"))).alias(
                f"{side}_tz_shift"
            ),
            (
                pl.lit(is_away)
                * (
                    1
                    + pl.when(pl.col(f"{side}_last_away") == 1)
                    .then(pl.col(f"{side}_last_trip"))
                    .otherwise(0)
                )
            )
            .cast(pl.Int64)
            .alias(f"{side}_road_trip_len"),
        ).drop(
            f"{side}_last_date",
            f"{side}_last_lat",
            f"{side}_last_lon",
            f"{side}_last_away",
            f"{side}_last_trip",
        )
    return base.drop("latitude", "longitude")


def _join_pitchers(base: pl.DataFrame, db: Database) -> pl.DataFrame:
    """Estado actual del pitcher probable: forma reciente + nivel previo."""
    apps = db.query(_APPEARANCES_SQL)
    savant = db.query(_SAVANT_PITCHER_SQL)

    last_app = apps.group_by("player_id").agg(
        pl.col("game_date").max().alias("_last_appearance")
    )
    starts = apps.filter(pl.col("is_starter")).sort("game_date", "game_pk")
    aggregations = [
        pl.col(c).tail(w).sum().alias(f"_{c}_sum_{w}") for w in WINDOWS for c in SUM_COLUMNS
    ] + [
        expr
        for w in WINDOWS
        for expr in (
            pl.col("pitches_thrown").tail(w).mean().alias(f"sp_pitches_avg_{w}"),
            pl.col("outs_recorded").tail(w).mean().alias(f"_outs_mean_{w}"),
        )
    ]
    state = (
        starts.group_by("player_id")
        .agg(aggregations)
        .with_columns(derived_expressions())
        .select("player_id", pl.selectors.starts_with("sp_"))
    )

    for side in ("home", "away"):
        pitcher_id = f"{side}_probable_pitcher_id"
        form = state.rename(
            {c: f"{side}_{c}" for c in state.columns if c != "player_id"}
        )
        base = base.join(
            form, left_on=pitcher_id, right_on="player_id", how="left"
        ).join(
            last_app.rename({"_last_appearance": f"{side}__last_appearance"}),
            left_on=pitcher_id,
            right_on="player_id",
            how="left",
        )
        base = base.with_columns(
            (pl.col("game_date") - pl.col(f"{side}__last_appearance"))
            .dt.total_days()
            .alias(f"{side}_sp_days_rest")
        ).drop(f"{side}__last_appearance")

        quality = savant.rename(
            {c: f"{side}_{c}" for c in savant.columns if c not in ("player_id", "year")}
        )
        base = base.join(
            quality.with_columns((pl.col("year") + 1).alias("_target_season")),
            left_on=[pitcher_id, "season"],
            right_on=["player_id", "_target_season"],
            how="left",
        ).drop("year")
    return base
