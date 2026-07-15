"""Fuente Baseball Savant: leaderboards de métricas Statcast agregadas.

Sustituye a FanGraphs como fuente de métricas avanzadas por temporada:
FanGraphs bloquea el acceso programático (Cloudflare) y no ofrece API
pública. Savant es de MLB, expone CSV directo y cubre las métricas clave:
xwOBA, xBA, xSLG, xERA, barrel%, hard hit%, exit velocity, launch angle,
whiff%, chase%, K%, BB%. FIP/xFIP/SIERA se calcularán en el feature
engineering a partir de datos propios.
"""

import io
import logging
from typing import Any, Literal

import pandas as pd
import requests

from mlb_quant.ingestion.base import DEFAULT_TIMEOUT, DataSource, build_session
from mlb_quant.utils.frames import sanitize_columns

logger = logging.getLogger(__name__)

EXPECTED_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
CUSTOM_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

#: Métricas del leaderboard custom (válidas para batter y pitcher).
STATCAST_METRICS: tuple[str, ...] = (
    "pa",
    "k_percent",
    "bb_percent",
    "woba",
    "xwoba",
    "babip",
    "barrel_batted_rate",
    "hard_hit_percent",
    "exit_velocity_avg",
    "launch_angle_avg",
    "sweet_spot_percent",
    "whiff_percent",
    "oz_swing_percent",
)

#: Clave única de una fila en las tablas Savant.
SAVANT_KEYS: tuple[str, ...] = ("player_id", "year", "player_type")

type PlayerType = Literal["batter", "pitcher"]


class SavantLeaderboards(DataSource):
    """Descarga leaderboards CSV de Baseball Savant."""

    source_name = "savant"

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Inicializa la fuente.

        Args:
            session: Sesión HTTP inyectable (para tests).
            timeout: Timeout por petición en segundos.
        """
        self._session = session or build_session()
        self._timeout = timeout

    def fetch_expected_stats(
        self, season: int, player_type: PlayerType, min_pa: int = 1
    ) -> pd.DataFrame:
        """Descarga expected statistics (xBA, xSLG, xwOBA; xERA para pitchers).

        Args:
            season: Año de la temporada.
            player_type: ``batter`` o ``pitcher``.
            min_pa: Mínimo de apariciones al plato / bateadores enfrentados.

        Returns:
            Una fila por jugador, con columna ``player_type`` añadida.
        """
        df = self._get_csv(
            EXPECTED_URL,
            params={
                "type": player_type,
                "year": season,
                "position": "",
                "team": "",
                "min": min_pa,
                "csv": "true",
            },
        )
        return self._clean(df, player_type, f"expected_stats {player_type} {season}")

    def fetch_statcast_metrics(
        self, season: int, player_type: PlayerType, min_pa: int = 1
    ) -> pd.DataFrame:
        """Descarga métricas Statcast agregadas (barrel%, EV, whiff%, chase%...).

        Args:
            season: Año de la temporada.
            player_type: ``batter`` o ``pitcher``.
            min_pa: Mínimo de apariciones al plato / bateadores enfrentados.

        Returns:
            Una fila por jugador, con columna ``player_type`` añadida.
        """
        df = self._get_csv(
            CUSTOM_URL,
            params={
                "year": season,
                "type": player_type,
                "filter": "",
                "min": min_pa,
                "selections": ",".join(STATCAST_METRICS),
                "chart": "false",
                "sort": "xwoba",
                "sortDir": "desc",
                "csv": "true",
            },
        )
        return self._clean(df, player_type, f"statcast_metrics {player_type} {season}")

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #

    def _get_csv(self, url: str, params: dict[str, Any]) -> pd.DataFrame:
        """GET que devuelve el CSV parseado."""
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        return pd.read_csv(io.BytesIO(response.content))

    def _clean(self, df: pd.DataFrame, player_type: str, label: str) -> pd.DataFrame:
        """Renombra, sanea, añade player_type y valida claves."""
        if df.empty:
            logger.info("Savant %s: sin datos.", label)
            return df
        df = df.rename(columns={"last_name, first_name": "name"})
        df = sanitize_columns(df)
        df["player_type"] = player_type
        missing = [k for k in SAVANT_KEYS if k not in df.columns]
        if missing:
            msg = f"Respuesta de Savant sin columnas clave: {missing}"
            raise ValueError(msg)
        df = df.drop_duplicates(subset=list(SAVANT_KEYS))
        logger.info("Savant %s: %d filas.", label, len(df))
        return df
