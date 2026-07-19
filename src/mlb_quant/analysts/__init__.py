"""Equipo de analistas determinísticos + cerebro pronosticador.

Punto de entrada: :func:`mlb_quant.analysts.brain.generate_daily_parlays`.
"""

from mlb_quant.analysts.base import AnalystReport, DailyContext
from mlb_quant.analysts.brain import DailyParlays, Parlay, generate_daily_parlays
from mlb_quant.analysts.game import GameAnalyst
from mlb_quant.analysts.players import PlayerAnalyst
from mlb_quant.analysts.weather import WeatherAnalyst

__all__ = [
    "AnalystReport",
    "DailyContext",
    "DailyParlays",
    "GameAnalyst",
    "Parlay",
    "PlayerAnalyst",
    "WeatherAnalyst",
    "generate_daily_parlays",
]
