"""Analista del clima: inclinación over/under por condiciones del juego.

Reglas físicas conocidas del beisbol, no aprendidas: calor y viento
hacia afuera aumentan carreras (la bola vuela más); frío y viento en
contra las reducen. Con techo (domo/retráctil) el clima se neutraliza.

La señal es una inclinación ``wx_total_lean`` en [-1, 1]: positivo
empuja al over, negativo al under, 0 = neutral o sin datos.
"""

import polars as pl

from mlb_quant.analysts.base import AnalystReport, DailyContext, empty_notes, has_columns

#: Umbrales de las reglas (unidades métricas, como el bloque de features).
HOT_C = 27.0
COLD_C = 10.0
WIND_OUT_KMH = 12.0

_HOT_LEAN = 0.3
_COLD_LEAN = -0.3
_WIND_LEAN = 0.4


class WeatherAnalyst:
    """Subagente de clima. Señales: ``wx_total_lean``, ``wx_has_data``."""

    name = "clima"

    def analyze(self, ctx: DailyContext) -> AnalystReport:
        """Calcula la inclinación al total por juego.

        Args:
            ctx: Contexto del día; usa ``wx_temperature_c``,
                ``wx_wind_out`` y ``wx_roof_possible`` si existen.

        Returns:
            Reporte con una fila por juego. Sin columnas de clima (bloque
            omitido en el builder) devuelve lean 0 y ``wx_has_data`` falso.
        """
        games = ctx.games
        if games.is_empty() or not has_columns(games, "game_pk"):
            return AnalystReport(self.name, pl.DataFrame(), empty_notes())

        if not has_columns(games, "wx_temperature_c", "wx_wind_out", "wx_roof_possible"):
            signals = games.select(
                "game_pk",
                pl.lit(0.0).alias("wx_total_lean"),
                pl.lit(False).alias("wx_has_data"),
            )
            return AnalystReport(self.name, signals, empty_notes())

        temp = pl.col("wx_temperature_c")
        wind_out = pl.col("wx_wind_out")
        roof = pl.col("wx_roof_possible") == 1
        has_data = temp.is_not_null() & wind_out.is_not_null()

        temp_lean = (
            pl.when(temp >= HOT_C)
            .then(_HOT_LEAN)
            .when(temp <= COLD_C)
            .then(_COLD_LEAN)
            .otherwise(0.0)
        )
        wind_lean = (
            pl.when(wind_out >= WIND_OUT_KMH)
            .then(_WIND_LEAN)
            .when(wind_out <= -WIND_OUT_KMH)
            .then(-_WIND_LEAN)
            .otherwise(0.0)
        )
        lean = (
            pl.when(roof | ~has_data)
            .then(0.0)
            .otherwise((temp_lean + wind_lean).clip(-1.0, 1.0))
        )

        signals = games.select(
            "game_pk",
            lean.alias("wx_total_lean"),
            (has_data & ~roof).alias("wx_has_data"),
        )
        notes = self._notes(games)
        return AnalystReport(self.name, signals, notes)

    @staticmethod
    def _notes(games: pl.DataFrame) -> pl.DataFrame:
        temp = pl.col("wx_temperature_c")
        wind_out = pl.col("wx_wind_out")
        roof = pl.col("wx_roof_possible") == 1
        note = (
            pl.when(roof)
            .then(pl.lit("techo posible: clima neutralizado"))
            .when(temp.is_null() | wind_out.is_null())
            .then(pl.lit(None))
            .when((temp >= HOT_C) & (wind_out >= WIND_OUT_KMH))
            .then(
                pl.format(
                    "calor ({}°C) y viento hacia afuera ({} km/h): favorece over",
                    temp.round(0),
                    wind_out.round(0),
                )
            )
            .when(wind_out >= WIND_OUT_KMH)
            .then(pl.format("viento hacia afuera ({} km/h): favorece over", wind_out.round(0)))
            .when(wind_out <= -WIND_OUT_KMH)
            .then(pl.format("viento en contra ({} km/h): favorece under", wind_out.round(0)))
            .when(temp >= HOT_C)
            .then(pl.format("calor ({}°C): favorece over", temp.round(0)))
            .when(temp <= COLD_C)
            .then(pl.format("frío ({}°C): favorece under", temp.round(0)))
            .otherwise(pl.lit(None))
        )
        return (
            games.select("game_pk", note.alias("note"))
            .drop_nulls("note")
            .cast({"game_pk": pl.Int64})
        )
