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

VENUES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "venue_id": pl.Int64,
    "season": pl.Int32,
    "name": pl.Utf8,
    "city": pl.Utf8,
    "state": pl.Utf8,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "elevation": pl.Float64,
    "azimuth_angle": pl.Float64,
    "roof_type": pl.Utf8,
    "turf_type": pl.Utf8,
    "capacity": pl.Int64,
    "left_line": pl.Int64,
    "left": pl.Int64,
    "left_center": pl.Int64,
    "center": pl.Int64,
    "right_center": pl.Int64,
    "right": pl.Int64,
    "right_line": pl.Int64,
}

LINEUPS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "team_id": pl.Int64,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "batting_order": pl.Int64,
    "position": pl.Utf8,
    "is_starting_lineup": pl.Boolean,
    "is_substitute": pl.Boolean,
}

PITCHING_LINES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "team_id": pl.Int64,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "is_starter": pl.Boolean,
    "innings_pitched": pl.Utf8,
    "outs_recorded": pl.Int64,
    "pitches_thrown": pl.Int64,
    "batters_faced": pl.Int64,
    "strike_outs": pl.Int64,
    "base_on_balls": pl.Int64,
    "hits": pl.Int64,
    "home_runs": pl.Int64,
    "runs": pl.Int64,
    "earned_runs": pl.Int64,
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

    def fetch_venues(self, season: int) -> pl.DataFrame:
        """Descarga los estadios con coordenadas, dimensiones y techo.

        Incluye ``azimuth_angle`` (orientación del campo), clave para
        interpretar la dirección del viento respecto al home plate.

        Args:
            season: Año de la temporada.

        Returns:
            Un estadio por fila, esquema :data:`VENUES_SCHEMA`.
        """
        payload = self._get(
            "venues",
            params={"sportId": 1, "season": season, "hydrate": "location,fieldInfo"},
        )
        rows = [self._parse_venue(venue, season) for venue in payload.get("venues", [])]
        logger.info("Venues %d: %d filas.", season, len(rows))
        return pl.DataFrame(rows, schema=VENUES_SCHEMA)

    def fetch_boxscore(self, game_pk: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Descarga el boxscore de un juego: lineups y líneas de pitcheo.

        Args:
            game_pk: Identificador del juego.

        Returns:
            Tupla ``(lineups, pitching_lines)``. Lineups: una fila por
            jugador con turno al bate. Pitching lines: una fila por pitcher
            usado (base de features de fatiga y descanso).
        """
        payload = self._get(f"game/{game_pk}/boxscore", params={})
        lineup_rows: list[dict[str, Any]] = []
        pitching_rows: list[dict[str, Any]] = []
        for side in ("home", "away"):
            team = payload.get("teams", {}).get(side, {})
            team_id = team.get("team", {}).get("id")
            pitcher_ids: list[int] = team.get("pitchers", [])
            for player in team.get("players", {}).values():
                lineup_row = self._parse_lineup_entry(player, game_pk, team_id)
                if lineup_row is not None:
                    lineup_rows.append(lineup_row)
                pitching_row = self._parse_pitching_line(
                    player, game_pk, team_id, pitcher_ids
                )
                if pitching_row is not None:
                    pitching_rows.append(pitching_row)
        return (
            pl.DataFrame(lineup_rows, schema=LINEUPS_SCHEMA),
            pl.DataFrame(pitching_rows, schema=PITCHING_LINES_SCHEMA),
        )

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
    def _parse_venue(venue: dict[str, Any], season: int) -> dict[str, Any]:
        """Aplana el JSON de un estadio al esquema de la tabla ``venues``."""
        location = venue.get("location", {})
        coords = location.get("defaultCoordinates", {})
        field = venue.get("fieldInfo", {})
        return {
            "venue_id": venue.get("id"),
            "season": season,
            "name": venue.get("name"),
            "city": location.get("city"),
            "state": location.get("stateAbbrev"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "elevation": location.get("elevation"),
            "azimuth_angle": location.get("azimuthAngle"),
            "roof_type": field.get("roofType"),
            "turf_type": field.get("turfType"),
            "capacity": field.get("capacity"),
            "left_line": field.get("leftLine"),
            "left": field.get("left"),
            "left_center": field.get("leftCenter"),
            "center": field.get("center"),
            "right_center": field.get("rightCenter"),
            "right": field.get("right"),
            "right_line": field.get("rightLine"),
        }

    @staticmethod
    def _parse_lineup_entry(
        player: dict[str, Any], game_pk: int, team_id: int | None
    ) -> dict[str, Any] | None:
        """Convierte un jugador del boxscore en fila de ``lineups``.

        ``battingOrder`` viene como string de 3 dígitos: ``"400"`` es el
        4to bate titular; ``"401"`` el primer sustituto de ese turno.
        Jugadores sin turno al bate (banca, pitchers con DH) se omiten.
        """
        raw_order = player.get("battingOrder")
        if not raw_order:
            return None
        order = int(raw_order)
        return {
            "game_pk": game_pk,
            "team_id": team_id,
            "player_id": player.get("person", {}).get("id"),
            "player_name": player.get("person", {}).get("fullName"),
            "batting_order": order // 100,
            "position": player.get("position", {}).get("abbreviation"),
            "is_starting_lineup": order % 100 == 0,
            "is_substitute": order % 100 != 0,
        }

    @staticmethod
    def _parse_pitching_line(
        player: dict[str, Any],
        game_pk: int,
        team_id: int | None,
        pitcher_ids: list[int],
    ) -> dict[str, Any] | None:
        """Convierte un jugador del boxscore en fila de ``pitching_lines``.

        Solo jugadores con stats de pitcheo en el juego. El abridor es el
        primer id de la lista ``pitchers`` del equipo.
        """
        stats = player.get("stats", {}).get("pitching", {})
        if not stats:
            return None
        player_id = player.get("person", {}).get("id")
        return {
            "game_pk": game_pk,
            "team_id": team_id,
            "player_id": player_id,
            "player_name": player.get("person", {}).get("fullName"),
            "is_starter": bool(pitcher_ids) and player_id == pitcher_ids[0],
            "innings_pitched": stats.get("inningsPitched"),
            "outs_recorded": stats.get("outs"),
            "pitches_thrown": stats.get("numberOfPitches"),
            "batters_faced": stats.get("battersFaced"),
            "strike_outs": stats.get("strikeOuts"),
            "base_on_balls": stats.get("baseOnBalls"),
            "hits": stats.get("hits"),
            "home_runs": stats.get("homeRuns"),
            "runs": stats.get("runs"),
            "earned_runs": stats.get("earnedRuns"),
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
