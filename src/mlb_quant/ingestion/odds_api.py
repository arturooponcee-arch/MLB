"""Fuente de cuotas: The Odds API (https://the-odds-api.com).

Cuotas decimales de MLB por casa (bookmaker) y mercado. Cada descarga es
un snapshot con marca de tiempo: acumular snapshots en DuckDB permite
medir CLV (closing line value) comparando la cuota tomada contra la del
cierre del mercado.

Fuente pura: recibe la API key, no consulta la base de datos.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import polars as pl
import requests

from mlb_quant.ingestion.base import DEFAULT_TIMEOUT, DataSource, build_session

logger = logging.getLogger(__name__)

ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

# Claves de upsert: un snapshot no pisa a los anteriores (histórico para CLV).
ODDS_KEYS = ("snapshot_utc", "event_id", "book", "market", "outcome")

ODDS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "snapshot_utc": pl.Utf8,
    "event_id": pl.Utf8,
    "commence_time_utc": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "book": pl.Utf8,
    "market": pl.Utf8,
    "outcome": pl.Utf8,
    "point": pl.Float64,
    "decimal_odds": pl.Float64,
    "last_update_utc": pl.Utf8,
}


class OddsApiSource(DataSource):
    """Descarga cuotas de MLB de The Odds API en formato largo (tidy)."""

    source_name = "the_odds_api"

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        now: datetime | None = None,
    ) -> None:
        """Inicializa la fuente.

        Args:
            api_key: API key de The Odds API (``ODDS_API_KEY`` en ``.env``).
            session: Sesión HTTP inyectable (para tests).
            timeout: Timeout por petición en segundos.
            now: Instante "ahora" inyectable (para tests del snapshot).

        Raises:
            ValueError: Si la API key está vacía.
        """
        if not api_key:
            msg = "ODDS_API_KEY vacía. Configúrala en .env (ver .env.example)."
            raise ValueError(msg)
        self._api_key = api_key
        self._session = session or build_session()
        self._timeout = timeout
        self._now = now

    def fetch_odds(
        self,
        markets: tuple[str, ...] = ("h2h", "totals"),
        regions: tuple[str, ...] = ("us",),
    ) -> pl.DataFrame:
        """Descarga las cuotas vigentes de los próximos juegos de MLB.

        Una petición cubre todos los eventos listados por la API (los
        juegos de hoy y próximos). El costo en cuota de la API es
        ``len(markets) x len(regions)`` por llamada.

        Args:
            markets: Mercados a pedir (``h2h`` = moneyline, ``totals``...).
            regions: Regiones de casas de apuestas (``us``, ``eu``...).

        Returns:
            Una fila por evento x casa x mercado x outcome, esquema
            :data:`ODDS_SCHEMA`. ``point`` solo aplica a totals/spreads.
        """
        payload = self._get(
            ODDS_URL,
            params={
                "apiKey": self._api_key,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        snapshot = (self._now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [
            {
                "snapshot_utc": snapshot,
                "event_id": event.get("id"),
                "commence_time_utc": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "book": bookmaker.get("key"),
                "market": market.get("key"),
                "outcome": outcome.get("name"),
                "point": outcome.get("point"),
                "decimal_odds": outcome.get("price"),
                "last_update_utc": market.get("last_update") or bookmaker.get("last_update"),
            }
            for event in payload
            for bookmaker in event.get("bookmakers", [])
            for market in bookmaker.get("markets", [])
            for outcome in market.get("outcomes", [])
        ]
        logger.info(
            "Cuotas: %d eventos, %d filas (mercados=%s).",
            len(payload),
            len(rows),
            ",".join(markets),
        )
        return pl.DataFrame(rows, schema=ODDS_SCHEMA)

    def _get(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET con reintentos; devuelve el JSON decodificado (lista de eventos)."""
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info("The Odds API: %s peticiones restantes en la cuota.", remaining)
        data: list[dict[str, Any]] = response.json()
        return data
