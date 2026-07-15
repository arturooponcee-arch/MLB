"""Fuente de clima: Open-Meteo (sin API key).

Clima horario por estadio: temperatura, humedad, viento (velocidad y
dirección), presión. Histórico vía archive API; hoy/futuro vía forecast API.

Fuente pura: recibe coordenadas, no consulta la base de datos.
"""

import logging
from datetime import date
from typing import Any

import polars as pl
import requests

from mlb_quant.ingestion.base import DEFAULT_TIMEOUT, DataSource, build_session

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)

WEATHER_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "venue_id": pl.Int64,
    "time_utc": pl.Utf8,
    "temperature_c": pl.Float64,
    "humidity_pct": pl.Float64,
    "wind_speed_kmh": pl.Float64,
    "wind_direction_deg": pl.Float64,
    "pressure_hpa": pl.Float64,
}


class WeatherSource(DataSource):
    """Descarga clima horario de Open-Meteo para unas coordenadas."""

    source_name = "open_meteo"

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        today: date | None = None,
    ) -> None:
        """Inicializa la fuente.

        Args:
            session: Sesión HTTP inyectable (para tests).
            timeout: Timeout por petición en segundos.
            today: Fecha "hoy" inyectable (para tests de selección de
                endpoint). Por defecto, la fecha actual.
        """
        self._session = session or build_session()
        self._timeout = timeout
        self._today = today or date.today()

    def fetch_hourly(
        self,
        venue_id: int,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Descarga clima horario para un estadio y rango de fechas.

        Usa la archive API para rangos completamente pasados (datos
        consolidados, disponibles con ~5 días de retraso) y la forecast API
        si el rango incluye hoy o fechas futuras.

        Args:
            venue_id: Id del estadio (solo se propaga a la salida).
            latitude: Latitud del estadio.
            longitude: Longitud del estadio.
            start_date: Primer día (inclusive).
            end_date: Último día (inclusive).

        Returns:
            Una fila por hora, esquema :data:`WEATHER_SCHEMA`.
        """
        url = ARCHIVE_URL if end_date < self._today else FORECAST_URL
        payload = self._get(
            url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(_HOURLY_VARS),
                "timezone": "UTC",
            },
        )
        hourly = payload.get("hourly", {})
        times: list[str] = hourly.get("time", [])
        rows = [
            {
                "venue_id": venue_id,
                "time_utc": times[i],
                "temperature_c": hourly.get("temperature_2m", [None] * len(times))[i],
                "humidity_pct": hourly.get("relative_humidity_2m", [None] * len(times))[i],
                "wind_speed_kmh": hourly.get("wind_speed_10m", [None] * len(times))[i],
                "wind_direction_deg": hourly.get("wind_direction_10m", [None] * len(times))[i],
                "pressure_hpa": hourly.get("surface_pressure", [None] * len(times))[i],
            }
            for i in range(len(times))
        ]
        logger.info(
            "Clima venue=%d %s -> %s: %d horas.", venue_id, start_date, end_date, len(rows)
        )
        return pl.DataFrame(rows, schema=WEATHER_SCHEMA)

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET con reintentos; devuelve el JSON decodificado."""
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data
