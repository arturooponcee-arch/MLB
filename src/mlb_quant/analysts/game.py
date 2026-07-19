"""Analista por juego: convierte las predicciones en piernas candidatas.

Para cada juego toma el lado más probable de cada mercado (moneyline,
total, run line, F5) y lo emite como pierna candidata con su
probabilidad del modelo. El cerebro decide después cuáles sobreviven y
cómo se combinan.
"""

import polars as pl

from mlb_quant.analysts.base import AnalystReport, DailyContext, empty_notes, has_columns

#: Línea fija del total que evalúa el simulador (ver SimulationConfig).
TOTAL_LINE = 8.5
RUN_LINE = 1.5


class GameAnalyst:
    """Subagente por juego. Salida principal: piernas candidatas."""

    name = "juego"

    def analyze(self, ctx: DailyContext) -> AnalystReport:
        """Extrae la pierna más probable de cada mercado por juego.

        Returns:
            Reporte cuyas ``signals`` son piernas: ``game_pk``, ``market``
            (``ml`` | ``total`` | ``runline`` | ``f5_ml``), ``selection``
            legible, ``side`` (``home``/``away``/``over``/``under``) y ``p``.
        """
        games = ctx.games
        needed = ("game_pk", "home_team_name", "away_team_name", "p_home_win")
        if games.is_empty() or not has_columns(games, *needed):
            return AnalystReport(self.name, pl.DataFrame(), empty_notes())

        legs = [self._moneyline(games)]
        if has_columns(games, "sim_p_over", "sim_p_under"):
            legs.append(self._total(games))
        if has_columns(games, "sim_p_home_runline", "sim_p_away_runline"):
            legs.append(self._runline(games))
        if has_columns(games, "sim_p_f5_home_ml", "sim_p_f5_tie"):
            legs.append(self._f5(games))

        signals = pl.concat(legs).with_columns(pl.col("game_pk").cast(pl.Int64))
        return AnalystReport(self.name, signals, empty_notes())

    @staticmethod
    def _moneyline(games: pl.DataFrame) -> pl.DataFrame:
        p_home = pl.col("p_home_win")
        home_wins = p_home >= 0.5
        return games.select(
            "game_pk",
            pl.lit("ml").alias("market"),
            pl.when(home_wins)
            .then(pl.format("ML: {} (local)", "home_team_name"))
            .otherwise(pl.format("ML: {} (visita)", "away_team_name"))
            .alias("selection"),
            pl.when(home_wins).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side"),
            pl.when(home_wins).then(p_home).otherwise(1.0 - p_home).alias("p"),
        )

    @staticmethod
    def _total(games: pl.DataFrame) -> pl.DataFrame:
        over = pl.col("sim_p_over")
        under = pl.col("sim_p_under")
        over_wins = over >= under
        matchup = pl.format("{} @ {}", "away_team_name", "home_team_name")
        return games.select(
            "game_pk",
            pl.lit("total").alias("market"),
            pl.when(over_wins)
            .then(pl.format("Over {}: {}", pl.lit(TOTAL_LINE), matchup))
            .otherwise(pl.format("Under {}: {}", pl.lit(TOTAL_LINE), matchup))
            .alias("selection"),
            pl.when(over_wins).then(pl.lit("over")).otherwise(pl.lit("under")).alias("side"),
            pl.when(over_wins).then(over).otherwise(under).alias("p"),
        )

    @staticmethod
    def _runline(games: pl.DataFrame) -> pl.DataFrame:
        home_rl = pl.col("sim_p_home_runline")
        away_rl = pl.col("sim_p_away_runline")
        home_wins = home_rl >= away_rl
        return games.select(
            "game_pk",
            pl.lit("runline").alias("market"),
            pl.when(home_wins)
            .then(pl.format("RL -{}: {}", pl.lit(RUN_LINE), "home_team_name"))
            .otherwise(pl.format("RL +{}: {}", pl.lit(RUN_LINE), "away_team_name"))
            .alias("selection"),
            pl.when(home_wins).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side"),
            pl.when(home_wins).then(home_rl).otherwise(away_rl).alias("p"),
        )

    @staticmethod
    def _f5(games: pl.DataFrame) -> pl.DataFrame:
        """F5 asumiendo mercado a 2 vías con empate = push.

        La probabilidad condicional a que no haya empate es la relevante
        para una cuota con push: p / (1 - p_tie).
        """
        p_home = pl.col("sim_p_f5_home_ml")
        p_tie = pl.col("sim_p_f5_tie")
        p_away = 1.0 - p_home - p_tie
        decided = 1.0 - p_tie
        home_wins = p_home >= p_away
        return games.select(
            "game_pk",
            pl.lit("f5_ml").alias("market"),
            pl.when(home_wins)
            .then(pl.format("F5 ML: {}", "home_team_name"))
            .otherwise(pl.format("F5 ML: {}", "away_team_name"))
            .alias("selection"),
            pl.when(home_wins).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side"),
            pl.when(home_wins).then(p_home / decided).otherwise(p_away / decided).alias("p"),
        )
