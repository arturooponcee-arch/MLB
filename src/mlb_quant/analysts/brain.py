"""Cerebro pronosticador: consolida a los analistas y arma combinadas.

Flujo: el analista por juego propone piernas candidatas; el de clima y
el de jugadores emiten señales de contexto. El cerebro cruza todo:

- Descarta piernas de total que contradicen al clima (over con viento en
  contra, under con calor y viento a favor).
- Descarta piernas F5 cuando la ventaja de abridores apunta al lado
  contrario (el F5 depende casi solo de los abridores).
- Premia en el ranking las piernas donde dos analistas coinciden.

De la piscina resultante arma 3 perfiles de combinada del día
(conservadora / equilibrada / agresiva) con una pierna por juego como
máximo — piernas del mismo juego están correlacionadas y la probabilidad
combinada dejaría de ser el producto.
"""

import logging
from dataclasses import dataclass

import polars as pl

from mlb_quant.analysts.base import DailyContext
from mlb_quant.analysts.game import GameAnalyst
from mlb_quant.analysts.players import PlayerAnalyst
from mlb_quant.analysts.weather import WeatherAnalyst

logger = logging.getLogger(__name__)

#: Perfiles: (nombre, piernas, probabilidad mínima por pierna).
PROFILES: tuple[tuple[str, int, float], ...] = (
    ("conservadora", 2, 0.62),
    ("equilibrada", 3, 0.58),
    ("agresiva", 4, 0.53),
)

#: |lean| de clima que ya cuenta como señal (no ruido).
_WX_MIN_LEAN = 0.2
#: |edge| de abridores que veta un F5 en contra.
_SP_MIN_EDGE = 0.25
#: Bono de score cuando dos analistas coinciden.
_AGREE_BONUS = 0.03


@dataclass(frozen=True)
class Parlay:
    """Una combinada sugerida.

    Attributes:
        name: Perfil (``conservadora`` | ``equilibrada`` | ``agresiva``).
        legs: Piernas elegidas (subconjunto de la piscina).
        p_combined: Producto de las probabilidades (juegos distintos,
            independencia asumida).
        fair_odds: Cuota justa combinada (1/p): solo tiene valor si el
            book paga más.
        substitute: Pierna suplente (mejor pierna no usada, de otro
            juego) cuando la combinada incluye props: no todos los books
            listan props de cada pitcher, así que el usuario puede
            sustituir sin rearmar. None si no hay props o no queda
            piscina.
    """

    name: str
    legs: pl.DataFrame
    p_combined: float
    fair_odds: float
    substitute: pl.DataFrame | None = None


@dataclass(frozen=True)
class DailyParlays:
    """Salida del cerebro: piscina de piernas, combinadas y notas."""

    pool: pl.DataFrame
    parlays: list[Parlay]
    notes: pl.DataFrame

    def to_frame(self) -> pl.DataFrame:
        """Aplana las combinadas para exportar (una fila por pierna)."""
        if not self.parlays:
            return pl.DataFrame()
        frames = [
            parlay.legs.select(
                pl.lit(parlay.name).alias("combinada"),
                "selection",
                "market",
                pl.col("p").round(3),
                pl.lit(round(parlay.p_combined, 3)).alias("p_combinada"),
                pl.lit(round(parlay.fair_odds, 2)).alias("cuota_justa"),
            )
            for parlay in self.parlays
        ]
        return pl.concat(frames)


def generate_daily_parlays(games: pl.DataFrame, props: pl.DataFrame | None = None) -> DailyParlays:
    """Corre a los analistas y arma las combinadas del día.

    Args:
        games: Salida de ``predict_upcoming_games``.
        props: Salida de ``predict_upcoming_strikeouts`` (opcional).

    Returns:
        Piscina de piernas ajustada, combinadas por perfil y notas de los
        analistas. Sin juegos: todo vacío.
    """
    ctx = DailyContext(games=games, props=props if props is not None else pl.DataFrame())
    empty = pl.DataFrame()
    if games.is_empty():
        return DailyParlays(pool=empty, parlays=[], notes=empty)

    weather = WeatherAnalyst().analyze(ctx)
    players = PlayerAnalyst().analyze(ctx)
    game = GameAnalyst().analyze(ctx)

    notes = pl.concat(
        [
            report.notes.with_columns(pl.lit(report.analyst).alias("analyst"))
            for report in (weather, players, game)
            if not report.notes.is_empty()
        ]
        or [pl.DataFrame(schema={"game_pk": pl.Int64, "note": pl.String, "analyst": pl.String})]
    )
    if not notes.is_empty() and {"home_team_name", "away_team_name"} <= set(games.columns):
        matchups = games.select(
            pl.col("game_pk").cast(pl.Int64),
            pl.format("{} @ {}", "away_team_name", "home_team_name").alias("matchup"),
        )
        notes = notes.join(matchups, on="game_pk", how="left")

    if game.signals.is_empty():
        return DailyParlays(pool=empty, parlays=[], notes=notes)

    pool = _adjust_pool(game.signals, weather.signals, players.signals)
    prop_legs = PlayerAnalyst.prop_legs(ctx)
    if not prop_legs.is_empty():
        pool = pl.concat(
            [
                pool,
                prop_legs.with_columns(
                    pl.lit("props").alias("side"), pl.col("p").alias("score")
                ).select(pool.columns),
            ]
        )

    parlays = [
        parlay
        for name, n_legs, min_p in PROFILES
        if (parlay := _build_parlay(pool, name, n_legs, min_p)) is not None
    ]
    logger.info(
        "Cerebro: %d piernas en piscina, %d combinadas generadas.", len(pool), len(parlays)
    )
    return DailyParlays(pool=pool, parlays=parlays, notes=notes)


def _adjust_pool(
    legs: pl.DataFrame, weather: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    """Aplica vetos y bonos de los analistas de contexto a las piernas."""
    pool = legs
    if not weather.is_empty():
        pool = pool.join(weather, on="game_pk", how="left")
    else:
        pool = pool.with_columns(
            pl.lit(0.0).alias("wx_total_lean"), pl.lit(False).alias("wx_has_data")
        )
    if not players.is_empty():
        pool = pool.join(players, on="game_pk", how="left")
    else:
        pool = pool.with_columns(
            pl.lit(0.0).alias("pitcher_edge_home"), pl.lit(False).alias("sp_has_data")
        )

    lean = pl.col("wx_total_lean").fill_null(0.0)
    edge = pl.col("pitcher_edge_home").fill_null(0.0)
    market = pl.col("market")
    side = pl.col("side")

    # Veto: total contra el clima.
    wx_conflict = (market == "total") & (
        ((side == "over") & (lean <= -_WX_MIN_LEAN))
        | ((side == "under") & (lean >= _WX_MIN_LEAN))
    )
    # Veto: F5 contra la ventaja de abridores.
    sp_conflict = (market == "f5_ml") & (
        ((side == "home") & (edge <= -_SP_MIN_EDGE))
        | ((side == "away") & (edge >= _SP_MIN_EDGE))
    )
    pool = pool.filter(~wx_conflict & ~sp_conflict)

    wx_agree = (market == "total") & (
        ((side == "over") & (lean >= _WX_MIN_LEAN))
        | ((side == "under") & (lean <= -_WX_MIN_LEAN))
    )
    sp_agree = (market.is_in(["f5_ml", "ml"])) & (
        ((side == "home") & (edge >= _SP_MIN_EDGE))
        | ((side == "away") & (edge <= -_SP_MIN_EDGE))
    )
    return pool.with_columns(
        (
            pl.col("p")
            + pl.when(wx_agree).then(_AGREE_BONUS).otherwise(0.0)
            + pl.when(sp_agree).then(_AGREE_BONUS).otherwise(0.0)
        ).alias("score")
    ).select("game_pk", "market", "selection", "side", "p", "score")


def _build_parlay(pool: pl.DataFrame, name: str, n_legs: int, min_p: float) -> Parlay | None:
    """Arma un perfil: mejores piernas por score, una por juego."""
    if pool.is_empty():
        return None
    candidates = (
        pool.filter(pl.col("p") >= min_p)
        .sort("score", descending=True)
        .unique(subset=["game_pk"], keep="first", maintain_order=True)
        .head(n_legs)
    )
    if len(candidates) < n_legs:
        return None
    p_combined = float(candidates["p"].product())
    if p_combined <= 0.0:
        return None

    substitute = None
    if (candidates["market"] == "prop_k").any():
        used_games = candidates["game_pk"].to_list()
        backups = (
            pool.filter(
                (pl.col("market") != "prop_k")
                & ~pl.col("game_pk").is_in(used_games)
                & (pl.col("p") >= min_p)
            )
            .sort("score", descending=True)
            .head(1)
        )
        substitute = backups if not backups.is_empty() else None

    return Parlay(
        name=name,
        legs=candidates,
        p_combined=p_combined,
        fair_odds=1.0 / p_combined,
        substitute=substitute,
    )
