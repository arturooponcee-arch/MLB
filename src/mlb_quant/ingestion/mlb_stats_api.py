"""Cliente de la MLB Stats API pública (statsapi.mlb.com).

Fuente de verdad para el "hoy": calendario, pitchers probables, marcadores
y equipos. Sin API key. Uso personal/analítico.
"""

import logging
from datetime import date
from typing import Any

import polars as pl
import requests

from mlb_quant.ingestion.base import DEFAULT_TIMEOUT, DataSource, build_session

logger = logging.getLogger(__name__)

SCHEDULE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "game_date": pl.Date,
    "game_datetime_utc": pl.Utf8,
    "season": pl.Int32,
    "game_type": pl.Utf8,
    "status": pl.Utf8,
    "double_header": pl.Utf8,
    "day_night": pl.Utf8,
    "venue_id": pl.Int64,
    "venue_name": pl.Utf8,
    "home_team_id": pl.Int64,
    "home_team_name": pl.Utf8,
    "away_team_id": pl.Int64,
    "away_team_name": pl.Utf8,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "home_probable_pitcher_id": pl.Int64,
    "home_probable_pitcher_name": pl.Utf8,
    "away_probable_pitcher_id": pl.Int64,
    "away_probable_pitcher_name": pl.Utf8,
}

TEAMS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "team_id": pl.Int64,
    "season": pl.Int32,
    "name": pl.Utf8,
    "abbreviation": pl.Utf8,
    "team_code": pl.Utf8,
    "league_id": pl.Int64,
    "league_name": pl.Utf8,
    "division_id": pl.Int64,
    "division_name": pl.Utf8,
    "venue_id": pl.Int64,
    "venue_name": pl.Utf8,
}


class MlbStatsApi(DataSource):
    """Acceso tipado a los endpoints públicos de la MLB Stats API."""

    source_name = "mlb_stats_api"
    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Inicializa el cliente.

        Args:
            session: Sesión HTTP inyectable (para tests). Por defecto se
                crea una con reintentos.
            timeout: Timeout por petición en segundos.
        """
        self._session = session or build_session()
        self._timeout = timeout

    def fetch_schedule(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Descarga el calendario de juegos con probables y marcadores.

        Args:
            start_date: Primer día del rango (inclusive).
            end_date: Último día del rango (inclusive).

        Returns:
            Un juego por fila, esquema :data:`SCHEDULE_SCHEMA`.
        """
        payload = self._get(
            "schedule",
            params={
                "sportId": 1,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "hydrate": "probablePitcher",
            },
        )
        rows = [
            self._parse_game(game)
            for day in payload.get("dates", [])
            for game in day.get("games", [])
        ]
        logger.info(
            "Calendario %s -> %s: %d juegos.", start_date, end_date, len(rows)
        )
        return pl.DataFrame(rows, schema=SCHEDULE_SCHEMA)

    def fetch_teams(self, season: int) -> pl.DataFrame:
        """Descarga los 30 equipos MLB de una temporada.

        Args:
            season: Año de la temporada.

        Returns:
            Un equipo por fila, esquema :data:`TEAMS_SCHEMA`.
        """
        payload = self._get("teams", params={"sportId": 1, "season": season})
        rows = [self._parse_team(team, season) for team in payload.get("teams", [])]
        logger.info("Equipos %d: %d filas.", season, len(rows))
        return pl.DataFrame(rows, schema=TEAMS_SCHEMA)

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET con reintentos; devuelve el JSON decodificado."""
        url = f"{self.BASE_URL}/{path}"
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    @staticmethod
    def _parse_game(game: dict[str, Any]) -> dict[str, Any]:
        """Aplana el JSON de un juego al esquema de la tabla ``games``."""
        home = game.get("teams", {}).get("home", {})
        away = game.get("teams", {}).get("away", {})
        home_pp = home.get("probablePitcher", {})
        away_pp = away.get("probablePitcher", {})
        return {
            "game_pk": game.get("gamePk"),
            "game_date": date.fromisoformat(game["officialDate"]),
            "game_datetime_utc": game.get("gameDate"),
            "season": int(game["season"]) if game.get("season") else None,
            "game_type": game.get("gameType"),
            "status": game.get("status", {}).get("detailedState"),
            "double_header": game.get("doubleHeader"),
            "day_night": game.get("dayNight"),
            "venue_id": game.get("venue", {}).get("id"),
            "venue_name": game.get("venue", {}).get("name"),
            "home_team_id": home.get("team", {}).get("id"),
            "home_team_name": home.get("team", {}).get("name"),
            "away_team_id": away.get("team", {}).get("id"),
            "away_team_name": away.get("team", {}).get("name"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "home_probable_pitcher_id": home_pp.get("id"),
            "home_probable_pitcher_name": home_pp.get("fullName"),
            "away_probable_pitcher_id": away_pp.get("id"),
            "away_probable_pitcher_name": away_pp.get("fullName"),
        }

    @staticmethod
    def _parse_team(team: dict[str, Any], season: int) -> dict[str, Any]:
        """Aplana el JSON de un equipo al esquema de la tabla ``teams``."""
        return {
            "team_id": team.get("id"),
            "season": season,
            "name": team.get("name"),
            "abbreviation": team.get("abbreviation"),
            "team_code": team.get("teamCode"),
            "league_id": team.get("league", {}).get("id"),
            "league_name": team.get("league", {}).get("name"),
            "division_id": team.get("division", {}).get("id"),
            "division_name": team.get("division", {}).get("name"),
            "venue_id": team.get("venue", {}).get("id"),
            "venue_name": team.get("venue", {}).get("name"),
        }
