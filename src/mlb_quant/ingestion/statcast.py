"""Fuente Statcast (Baseball Savant) vía pybaseball.

Datos pitch-by-pitch: exit velocity, launch angle, spin rate, resultado de
cada lanzamiento. Base histórica para el feature engineering.

Requiere el grupo opcional de dependencias ``data``:
``poetry install --with data``.
"""

import logging
from datetime import date

import pandas as pd

from mlb_quant.ingestion.base import DataSource

logger = logging.getLogger(__name__)

#: Clave única de un lanzamiento en la tabla ``statcast_pitches``.
PITCH_KEYS: tuple[str, ...] = ("game_pk", "at_bat_number", "pitch_number")


class StatcastSource(DataSource):
    """Descarga pitch-by-pitch de Baseball Savant."""

    source_name = "statcast"

    def fetch_pitches(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Descarga todos los lanzamientos de un rango de fechas.

        pybaseball trocea internamente el rango y cachea peticiones.
        Volumen aproximado: ~4.500 lanzamientos por día de temporada.

        Args:
            start_date: Primer día (inclusive).
            end_date: Último día (inclusive).

        Returns:
            Un lanzamiento por fila, columnas crudas de Statcast.

        Raises:
            ValueError: Si faltan las columnas clave en la respuesta.
        """
        df = self._download(start_date, end_date)
        if df.empty:
            logger.info("Statcast %s -> %s: sin datos.", start_date, end_date)
            return df
        missing = [k for k in PITCH_KEYS if k not in df.columns]
        if missing:
            msg = f"Respuesta de Statcast sin columnas clave: {missing}"
            raise ValueError(msg)
        df = df.drop_duplicates(subset=list(PITCH_KEYS))
        logger.info(
            "Statcast %s -> %s: %d lanzamientos.", start_date, end_date, len(df)
        )
        return df

    def _download(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Llama a pybaseball (import perezoso: dependencia opcional)."""
        from pybaseball import statcast

        result: pd.DataFrame = statcast(
            start_dt=start_date.isoformat(),
            end_dt=end_date.isoformat(),
            verbose=False,
        )
        return result
