"""Bloque de rating Elo por equipo.

Rating secuencial estilo FiveThirtyEight: base 1500, K = 4, ventaja de
local de +24 puntos aplicada solo dentro de la expectativa (no altera el
rating guardado) y regresión de 1/3 hacia 1500 entre temporadas. Sin
multiplicador por margen de victoria: en MLB aporta poco y añade un
knob de ajuste.

Anti-fuga por construcción: cada fila emite el rating ANTES de
actualizarlo con el resultado del propio juego.
"""

from typing import ClassVar

import polars as pl

from mlb_quant.db import Database
from mlb_quant.feature_engineering.base import FeatureBlock, Grain, register

BASE_RATING = 1500.0
K_FACTOR = 4.0
HOME_ADVANTAGE = 24.0
#: Fracción del rating (sobre la base) que se conserva entre temporadas.
SEASON_CARRY = 2.0 / 3.0

ELO_GAMES_SQL = """
SELECT game_pk, game_date, season, home_team_id, away_team_id,
       home_score, away_score
FROM games
WHERE status = 'Final' AND game_type = 'R'
ORDER BY game_date, game_pk
"""

_ELO_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "team_id": pl.Int64,
    "elo_pre": pl.Float64,
}


def compute_elo(games: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, float]]:
    """Recorre los juegos en orden y calcula el Elo pre-juego de cada equipo.

    Args:
        games: Juegos finalizados ordenados por ``(game_date, game_pk)``,
            con ids de equipos, marcador y temporada.

    Returns:
        Tupla ``(frame, ratings)``: una fila por ``(game_pk, team_id)`` con
        ``elo_pre`` (rating antes de ese juego), y el rating vigente por
        equipo tras el último juego (estado "as of" para juegos futuros).
    """
    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}
    rows: list[dict[str, float | int]] = []

    for game in games.iter_rows(named=True):
        home, away = game["home_team_id"], game["away_team_id"]
        for team in (home, away):
            rating = ratings.get(team, BASE_RATING)
            if last_season.get(team, game["season"]) != game["season"]:
                rating = BASE_RATING + SEASON_CARRY * (rating - BASE_RATING)
            ratings[team] = rating
            last_season[team] = game["season"]

        rows.append({"game_pk": game["game_pk"], "team_id": home, "elo_pre": ratings[home]})
        rows.append({"game_pk": game["game_pk"], "team_id": away, "elo_pre": ratings[away]})

        expected_home = 1.0 / (
            1.0 + 10.0 ** (-((ratings[home] + HOME_ADVANTAGE) - ratings[away]) / 400.0)
        )
        if game["home_score"] > game["away_score"]:
            actual_home = 1.0
        elif game["home_score"] < game["away_score"]:
            actual_home = 0.0
        else:
            actual_home = 0.5
        delta = K_FACTOR * (actual_home - expected_home)
        ratings[home] += delta
        ratings[away] -= delta

    return pl.DataFrame(rows, schema=_ELO_SCHEMA), ratings


@register
class EloBlock(FeatureBlock):
    """Rating Elo pre-juego por equipo.

    Salida por ``(game_pk, team_id)``: ``elo_pre``.
    """

    name: ClassVar[str] = "elo"
    grain: ClassVar[Grain] = "team_game"
    requires: ClassVar[tuple[str, ...]] = ("games",)

    def build(self, db: Database) -> pl.DataFrame:
        """Calcula el Elo de cada equipo antes de cada juego."""
        frame, _ = compute_elo(db.query(ELO_GAMES_SQL))
        return frame
