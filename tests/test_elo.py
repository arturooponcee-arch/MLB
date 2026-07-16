"""Tests del bloque Elo (feature_engineering.blocks.elo)."""

from datetime import date

import polars as pl
import pytest

from mlb_quant.db import Database
from mlb_quant.feature_engineering.blocks.elo import (
    BASE_RATING,
    HOME_ADVANTAGE,
    K_FACTOR,
    SEASON_CARRY,
    EloBlock,
    compute_elo,
)


def _games(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "game_pk": pl.Int64,
            "game_date": pl.Date,
            "season": pl.Int32,
            "home_team_id": pl.Int64,
            "away_team_id": pl.Int64,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
        },
        orient="row",
    )


def _expected_home(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((r_home + HOME_ADVANTAGE) - r_away) / 400.0))


def test_first_game_starts_at_base() -> None:
    frame, _ = compute_elo(_games([(1, date(2024, 4, 1), 2024, 10, 20, 5, 3)]))
    assert frame["elo_pre"].to_list() == [BASE_RATING, BASE_RATING]


def test_winner_gains_k_times_surprise() -> None:
    games = _games(
        [
            (1, date(2024, 4, 1), 2024, 10, 20, 5, 3),
            (2, date(2024, 4, 2), 2024, 10, 20, 2, 1),
        ]
    )
    frame, _ = compute_elo(games)
    delta = K_FACTOR * (1.0 - _expected_home(BASE_RATING, BASE_RATING))
    second = frame.filter(pl.col("game_pk") == 2)
    assert second.filter(pl.col("team_id") == 10)["elo_pre"][0] == pytest.approx(
        BASE_RATING + delta
    )
    assert second.filter(pl.col("team_id") == 20)["elo_pre"][0] == pytest.approx(
        BASE_RATING - delta
    )


def test_ratings_are_zero_sum() -> None:
    games = _games(
        [
            (1, date(2024, 4, 1), 2024, 10, 20, 5, 3),
            (2, date(2024, 4, 2), 2024, 20, 10, 7, 1),
            (3, date(2024, 4, 3), 2024, 10, 20, 0, 2),
        ]
    )
    _, ratings = compute_elo(games)
    assert sum(ratings.values()) == pytest.approx(2 * BASE_RATING)


def test_season_boundary_regresses_toward_base() -> None:
    games_2023 = [
        (1, date(2023, 6, 1), 2023, 10, 20, 5, 3),
        (2, date(2023, 6, 2), 2023, 10, 20, 4, 2),
    ]
    _, end_2023 = compute_elo(_games(games_2023))

    full = _games([*games_2023, (3, date(2024, 4, 1), 2024, 10, 20, 1, 0)])
    frame, _ = compute_elo(full)
    opener = frame.filter(pl.col("game_pk") == 3)
    for team in (10, 20):
        expected = BASE_RATING + SEASON_CARRY * (end_2023[team] - BASE_RATING)
        assert opener.filter(pl.col("team_id") == team)["elo_pre"][0] == pytest.approx(
            expected
        )


def test_tie_moves_toward_expectation() -> None:
    _, ratings = compute_elo(_games([(1, date(2024, 4, 1), 2024, 10, 20, 3, 3)]))
    # Empate: el local (favorito por +24) pierde rating, el visitante gana.
    assert ratings[10] < BASE_RATING < ratings[20]


def test_block_emits_one_row_per_team_game(warehouse: Database) -> None:
    result = EloBlock().build(warehouse)
    assert set(result.columns) == {"game_pk", "team_id", "elo_pre"}
    # 5 juegos finalizados x 2 equipos.
    assert len(result) == 10
    assert result.select("game_pk", "team_id").n_unique() == 10


def test_block_winner_rating_grows(warehouse: Database) -> None:
    # El equipo 10 gana todo: su elo_pre debe ser creciente juego a juego.
    result = EloBlock().build(warehouse).filter(pl.col("team_id") == 10)
    ratings = (
        result.join(
            warehouse.query("SELECT game_pk, game_date FROM games"), on="game_pk"
        )
        .sort("game_date")["elo_pre"]
        .to_list()
    )
    assert ratings == sorted(ratings)
    assert ratings[0] == BASE_RATING
