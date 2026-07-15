"""Bloque de clima: condiciones a la hora del juego + viento relativo.

El viento se descompone respecto al azimuth del estadio (orientación
home plate -> center field): componente "out" positivo = viento soplando
hacia el jardín central (favorece HR), componente "cross" = lateral.
"""

import math
from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

_WEATHER_SQL = """
SELECT g.game_pk,
       w.temperature_c AS wx_temperature_c,
       w.humidity_pct AS wx_humidity_pct,
       w.wind_speed_kmh AS wx_wind_speed_kmh,
       w.wind_direction_deg AS _wind_direction_deg,
       w.pressure_hpa AS wx_pressure_hpa,
       v.azimuth_angle AS _azimuth,
       CAST(v.roof_type IN ('Dome', 'Retractable') AS INTEGER) AS wx_roof_possible
FROM games g
LEFT JOIN venues v ON g.venue_id = v.venue_id AND v.season = g.season
LEFT JOIN weather_hourly w
    ON w.venue_id = g.venue_id
    AND w.time_utc = strftime(
        date_trunc('hour', CAST(g.game_datetime_utc AS TIMESTAMP)),
        '%Y-%m-%dT%H:%M'
    )
"""


@register
class WeatherBlock(FeatureBlock):
    """Clima en la hora de inicio del juego.

    Salida por ``game_pk``: temperatura, humedad, presión, velocidad del
    viento y componentes ``wx_wind_out`` / ``wx_wind_cross`` relativos a
    la orientación del estadio. Null si no hay clima ingerido para esa
    hora (correr ``mlb ingest weather``).
    """

    name: ClassVar[str] = "weather"
    grain: ClassVar[Grain] = "game"
    requires: ClassVar[tuple[str, ...]] = ("games", "venues", "weather_hourly")

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula features de clima para cada juego."""
        df = db.query(_WEATHER_SQL)

        # Meteorología reporta de DÓNDE viene el viento; hacia dónde va es
        # +180°. Out = proyección sobre el eje home->center (azimuth).
        blowing_to = (pl.col("_wind_direction_deg") + 180.0) * math.pi / 180.0
        azimuth = pl.col("_azimuth") * math.pi / 180.0
        angle = blowing_to - azimuth
        df = df.with_columns(
            (pl.col("wx_wind_speed_kmh") * angle.cos()).alias("wx_wind_out"),
            (pl.col("wx_wind_speed_kmh") * angle.sin()).alias("wx_wind_cross"),
        )
        return df.drop("_wind_direction_deg", "_azimuth")
