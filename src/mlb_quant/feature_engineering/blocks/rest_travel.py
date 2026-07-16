"""Bloque de descanso y viaje del equipo.

Descanso en días, kilómetros viajados desde el estadio anterior
(haversine), cambio de huso horario (aproximado por longitud, sin
dependencia de timezonefinder) y longitud de la gira actual como
visitante.

Sin fuga: el estadio y la fecha del propio juego son información de
calendario conocida antes del primer pitcheo; lo único histórico es el
juego anterior del equipo.
"""

import math
from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register
from mlb_quant.feature_engineering.rolling import add_days_since_previous

EARTH_RADIUS_KM = 6371.0

_LONG_SQL = """
SELECT game_pk, game_date, venue_id, home_team_id AS team_id, 0 AS is_away
FROM games
WHERE status = 'Final' AND game_type = 'R'
UNION ALL
SELECT game_pk, game_date, venue_id, away_team_id AS team_id, 1 AS is_away
FROM games
WHERE status = 'Final' AND game_type = 'R'
"""

#: Coordenadas por estadio (una fila por venue: la temporada más reciente).
VENUE_COORDS_SQL = """
SELECT venue_id, latitude, longitude
FROM (
    SELECT venue_id, latitude, longitude,
           ROW_NUMBER() OVER (PARTITION BY venue_id ORDER BY season DESC) AS rn
    FROM venues
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
)
WHERE rn = 1
"""


def haversine_km(
    lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr
) -> pl.Expr:
    """Distancia de círculo máximo en km entre dos coordenadas (grados)."""
    to_rad = math.pi / 180.0
    half_dlat = (lat2 - lat1) * (to_rad / 2.0)
    half_dlon = (lon2 - lon1) * (to_rad / 2.0)
    cos_product = (lat1 * to_rad).cos() * (lat2 * to_rad).cos()
    a = half_dlat.sin().pow(2) + cos_product * half_dlon.sin().pow(2)
    return 2.0 * EARTH_RADIUS_KM * a.sqrt().arcsin()


def tz_offset(longitude: pl.Expr) -> pl.Expr:
    """Huso horario aproximado (horas) a partir de la longitud.

    ``round(longitud / 15)``: suficiente como feature de desfase de
    reloj; evita la dependencia de una base de datos de husos reales.
    """
    return (longitude / 15.0).round(0)


def team_travel_history(db: Database) -> pl.DataFrame:
    """Historial por equipo con descanso, viaje, huso y gira calculados.

    Una fila por ``(game_pk, team_id)`` de juegos finalizados, ordenada
    por equipo y fecha, con las columnas de features del bloque ya
    derivadas. Compartida entre el builder histórico y el camino de
    juegos futuros (``upcoming``) para que las fórmulas sean idénticas.

    Args:
        db: Warehouse con ``games`` y ``venues``.

    Returns:
        Historial largo con features + columnas auxiliares (``is_away``,
        ``game_date``, coordenadas) que cada consumidor recorta.
    """
    long = (
        db.query(_LONG_SQL)
        .join(db.query(VENUE_COORDS_SQL), on="venue_id", how="left")
        .sort("team_id", "game_date", "game_pk")
    )
    long = add_days_since_previous(
        long, group_by="team_id", date_column="game_date", alias="team_days_rest"
    )
    long = long.with_columns(
        pl.col("latitude").shift(1).over("team_id").alias("_prev_lat"),
        pl.col("longitude").shift(1).over("team_id").alias("_prev_lon"),
        pl.col("is_away").shift(1).over("team_id").alias("_prev_away"),
    )
    long = long.with_columns(
        haversine_km(
            pl.col("_prev_lat"), pl.col("_prev_lon"), pl.col("latitude"), pl.col("longitude")
        ).alias("travel_km"),
        (tz_offset(pl.col("longitude")) - tz_offset(pl.col("_prev_lon"))).alias("tz_shift"),
        (pl.col("is_away") != pl.col("_prev_away"))
        .fill_null(value=True)
        .cast(pl.Int32)
        .cum_sum()
        .over("team_id")
        .alias("_run_id"),
    )
    # Posición dentro de la racha: en giras es 1, 2, 3...; en casa queda 0
    # porque is_away vale 0 en toda la racha local.
    long = long.with_columns(
        pl.col("is_away")
        .cum_sum()
        .over("team_id", "_run_id")
        .cast(pl.Int64)
        .alias("road_trip_len")
    )
    return long.drop("_prev_lat", "_prev_lon", "_prev_away", "_run_id")


@register
class RestTravelBlock(FeatureBlock):
    """Descanso, viaje, huso y gira del equipo antes de cada juego.

    Salida por ``(game_pk, team_id)``: ``team_days_rest``,
    ``road_trip_len``, ``travel_km``, ``tz_shift``.
    """

    name: ClassVar[str] = "rest_travel"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("games", "venues")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula las features de calendario de cada equipo."""
        return team_travel_history(db).select(
            "game_pk", "team_id", "team_days_rest", "road_trip_len", "travel_km", "tz_shift"
        )
