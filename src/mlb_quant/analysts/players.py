"""Analista de jugadores: ventaja de abridores y props con convicción.

Dos salidas:

- Señal ``pitcher_edge_home`` en [-1, 1] por juego: positivo = el abridor
  local es mejor que el visitante (por xERA de la temporada previa,
  Savant). Sirve para validar piernas que dependen del pitcheo (F5, ML).
- Piernas candidatas de props de strikeouts cuando el modelo tiene
  convicción clara (over o under).
"""

import polars as pl

from mlb_quant.analysts.base import AnalystReport, DailyContext, empty_notes, has_columns

#: xERA típica va de ~2.5 (élite) a ~6 (malo): 2 carreras de diferencia
#: entre abridores ya es una brecha enorme -> satura la señal.
_XERA_SPAN = 2.0

#: Convicción mínima del modelo de props para proponer la pierna.
PROP_MIN_P = 0.62


class PlayerAnalyst:
    """Subagente de jugadores. Señales: ``pitcher_edge_home``, ``sp_has_data``."""

    name = "jugadores"

    def analyze(self, ctx: DailyContext) -> AnalystReport:
        """Evalúa la ventaja de abridores por juego.

        Args:
            ctx: Contexto del día; usa ``home_spq_xera`` / ``away_spq_xera``
                si existen.

        Returns:
            Reporte con una fila por juego; edge 0 sin datos de Savant.
        """
        games = ctx.games
        if games.is_empty() or not has_columns(games, "game_pk"):
            return AnalystReport(self.name, pl.DataFrame(), empty_notes())

        if not has_columns(games, "home_spq_xera", "away_spq_xera"):
            signals = games.select(
                "game_pk",
                pl.lit(0.0).alias("pitcher_edge_home"),
                pl.lit(False).alias("sp_has_data"),
            )
            return AnalystReport(self.name, signals, empty_notes())

        home = pl.col("home_spq_xera")
        away = pl.col("away_spq_xera")
        has_data = home.is_not_null() & away.is_not_null()
        edge = (
            pl.when(has_data)
            .then(((away - home) / _XERA_SPAN).clip(-1.0, 1.0))
            .otherwise(0.0)
        )
        signals = games.select(
            "game_pk",
            edge.alias("pitcher_edge_home"),
            has_data.alias("sp_has_data"),
        )
        notes = self._notes(games)
        return AnalystReport(self.name, signals, notes)

    @staticmethod
    def prop_legs(ctx: DailyContext) -> pl.DataFrame:
        """Piernas candidatas de props de K con convicción del modelo.

        Returns:
            Filas ``game_pk``, ``market``, ``selection``, ``p`` — over si
            ``p_over >= PROP_MIN_P``, under si ``p_over <= 1 - PROP_MIN_P``.
            Vacío sin props.
        """
        props = ctx.props
        needed = ("game_pk", "pitcher_name", "line", "p_over")
        if props.is_empty() or not has_columns(props, *needed):
            return pl.DataFrame()
        overs = props.filter(pl.col("p_over") >= PROP_MIN_P).select(
            pl.col("game_pk").cast(pl.Int64),
            pl.lit("prop_k").alias("market"),
            pl.format("K Over {}: {}", "line", "pitcher_name").alias("selection"),
            pl.col("p_over").alias("p"),
        )
        unders = props.filter(pl.col("p_over") <= 1.0 - PROP_MIN_P).select(
            pl.col("game_pk").cast(pl.Int64),
            pl.lit("prop_k").alias("market"),
            pl.format("K Under {}: {}", "line", "pitcher_name").alias("selection"),
            (1.0 - pl.col("p_over")).alias("p"),
        )
        return pl.concat([overs, unders])

    @staticmethod
    def _notes(games: pl.DataFrame) -> pl.DataFrame:
        home = pl.col("home_spq_xera")
        away = pl.col("away_spq_xera")
        gap = away - home
        name_cols = has_columns(
            games, "home_probable_pitcher_name", "away_probable_pitcher_name"
        )
        if not name_cols:
            return empty_notes()
        note = (
            pl.when(home.is_null() | away.is_null())
            .then(pl.lit(None))
            .when(gap >= 1.0)
            .then(
                pl.format(
                    "ventaja clara del abridor local {} (xERA {} vs {})",
                    "home_probable_pitcher_name",
                    home.round(2),
                    away.round(2),
                )
            )
            .when(gap <= -1.0)
            .then(
                pl.format(
                    "ventaja clara del abridor visitante {} (xERA {} vs {})",
                    "away_probable_pitcher_name",
                    away.round(2),
                    home.round(2),
                )
            )
            .otherwise(pl.lit(None))
        )
        return (
            games.select("game_pk", note.alias("note"))
            .drop_nulls("note")
            .cast({"game_pk": pl.Int64})
        )
