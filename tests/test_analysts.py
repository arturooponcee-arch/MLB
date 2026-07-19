"""Tests del equipo de analistas y el cerebro de combinadas."""

import polars as pl
import pytest

from mlb_quant.analysts.base import DailyContext
from mlb_quant.analysts.brain import generate_daily_parlays
from mlb_quant.analysts.game import GameAnalyst
from mlb_quant.analysts.players import PlayerAnalyst
from mlb_quant.analysts.weather import WeatherAnalyst


def _game_row(
    game_pk: int,
    p_home_win: float = 0.55,
    p_over: float = 0.5,
    temp: float | None = 20.0,
    wind_out: float | None = 0.0,
    roof: int = 0,
    home_xera: float | None = 4.0,
    away_xera: float | None = 4.0,
) -> dict:
    return {
        "game_pk": game_pk,
        "home_team_name": f"Local{game_pk}",
        "away_team_name": f"Visita{game_pk}",
        "home_probable_pitcher_name": f"PitcherL{game_pk}",
        "away_probable_pitcher_name": f"PitcherV{game_pk}",
        "p_home_win": p_home_win,
        "sim_p_over": p_over,
        "sim_p_under": 1.0 - p_over,
        "sim_p_home_runline": 0.4,
        "sim_p_away_runline": 0.6,
        "sim_p_f5_home_ml": 0.5,
        "sim_p_f5_tie": 0.1,
        "wx_temperature_c": temp,
        "wx_wind_out": wind_out,
        "wx_roof_possible": roof,
        "home_spq_xera": home_xera,
        "away_spq_xera": away_xera,
    }


def _games(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def _ctx(*rows: dict) -> DailyContext:
    return DailyContext(games=_games(*rows))


def test_weather_wind_out_leans_over() -> None:
    report = WeatherAnalyst().analyze(_ctx(_game_row(1, wind_out=20.0)))
    assert report.signals["wx_total_lean"][0] > 0
    assert report.signals["wx_has_data"][0]


def test_weather_roof_neutralizes() -> None:
    report = WeatherAnalyst().analyze(_ctx(_game_row(1, wind_out=20.0, temp=30.0, roof=1)))
    assert report.signals["wx_total_lean"][0] == 0.0
    assert not report.signals["wx_has_data"][0]


def test_weather_missing_columns_is_neutral() -> None:
    games = _games(_game_row(1)).drop("wx_temperature_c", "wx_wind_out", "wx_roof_possible")
    report = WeatherAnalyst().analyze(DailyContext(games=games))
    assert report.signals["wx_total_lean"][0] == 0.0
    assert not report.signals["wx_has_data"][0]


def test_player_edge_sign_follows_xera_gap() -> None:
    report = PlayerAnalyst().analyze(_ctx(_game_row(1, home_xera=3.0, away_xera=5.0)))
    assert report.signals["pitcher_edge_home"][0] == pytest.approx(1.0)
    report = PlayerAnalyst().analyze(_ctx(_game_row(1, home_xera=5.0, away_xera=3.0)))
    assert report.signals["pitcher_edge_home"][0] == pytest.approx(-1.0)


def test_player_prop_legs_thresholds() -> None:
    props = pl.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "pitcher_name": ["A", "B", "C"],
            "line": [5.5, 5.5, 5.5],
            "p_over": [0.70, 0.50, 0.30],
        }
    )
    legs = PlayerAnalyst.prop_legs(DailyContext(games=pl.DataFrame(), props=props))
    assert len(legs) == 2  # over de A y under de C; B sin convicción
    assert legs.filter(pl.col("game_pk") == 3)["p"][0] == pytest.approx(0.70)


def test_player_prop_legs_include_team_name() -> None:
    props = pl.DataFrame(
        {
            "game_pk": [1],
            "pitcher_name": ["Jacob Lopez"],
            "team_name": ["Athletics"],
            "line": [5.5],
            "p_over": [0.70],
        }
    )
    legs = PlayerAnalyst.prop_legs(DailyContext(games=pl.DataFrame(), props=props))
    assert legs["selection"][0] == "K Over 5.5: Jacob Lopez (Athletics)"


def test_brain_parlay_with_prop_gets_substitute() -> None:
    games = _games(
        *[_game_row(pk, p_home_win=0.70, p_over=0.68) for pk in (1, 2, 3, 4, 5)]
    )
    props = pl.DataFrame(
        {
            "game_pk": [1],
            "pitcher_name": ["Jacob Lopez"],
            "team_name": ["Athletics"],
            "line": [5.5],
            "p_over": [0.99],  # gana el ranking: entra a todas las combinadas
        }
    )
    result = generate_daily_parlays(games, props)
    for parlay in result.parlays:
        has_prop = (parlay.legs["market"] == "prop_k").any()
        if has_prop:
            assert parlay.substitute is not None
            sub = parlay.substitute.row(0, named=True)
            assert sub["market"] != "prop_k"
            assert sub["game_pk"] not in parlay.legs["game_pk"].to_list()


def test_game_analyst_emits_one_leg_per_market() -> None:
    report = GameAnalyst().analyze(_ctx(_game_row(1, p_home_win=0.65, p_over=0.60)))
    legs = report.signals
    assert set(legs["market"].to_list()) == {"ml", "total", "runline", "f5_ml"}
    ml = legs.filter(pl.col("market") == "ml")
    assert ml["side"][0] == "home"
    assert ml["p"][0] == pytest.approx(0.65)
    total = legs.filter(pl.col("market") == "total")
    assert total["side"][0] == "over"


def test_brain_vetoes_over_against_wind_in() -> None:
    games = _games(_game_row(1, p_over=0.70, wind_out=-20.0))
    result = generate_daily_parlays(games)
    totals = result.pool.filter(pl.col("market") == "total")
    assert totals.is_empty()


def test_brain_keeps_over_with_supporting_wind() -> None:
    games = _games(_game_row(1, p_over=0.70, wind_out=20.0))
    result = generate_daily_parlays(games)
    totals = result.pool.filter(pl.col("market") == "total")
    assert len(totals) == 1
    assert totals["score"][0] > totals["p"][0]  # bono por coincidencia


def test_brain_builds_parlays_one_leg_per_game() -> None:
    games = _games(
        *[_game_row(pk, p_home_win=0.70, p_over=0.68) for pk in (1, 2, 3, 4)]
    )
    result = generate_daily_parlays(games)
    names = [parlay.name for parlay in result.parlays]
    assert "conservadora" in names
    for parlay in result.parlays:
        assert parlay.legs["game_pk"].n_unique() == len(parlay.legs)
        expected = float(parlay.legs["p"].product())
        assert parlay.p_combined == pytest.approx(expected)
        assert parlay.fair_odds == pytest.approx(1.0 / expected)


def test_brain_no_games_no_parlays() -> None:
    result = generate_daily_parlays(pl.DataFrame())
    assert result.parlays == []
    assert result.pool.is_empty()


def test_to_frame_flattens_parlays() -> None:
    games = _games(
        *[_game_row(pk, p_home_win=0.70, p_over=0.68) for pk in (1, 2, 3, 4)]
    )
    frame = generate_daily_parlays(games).to_frame()
    assert {"combinada", "selection", "market", "p", "p_combinada", "cuota_justa"} <= set(
        frame.columns
    )
    assert not frame.is_empty()
