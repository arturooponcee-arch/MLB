"""Ledger de apuestas: registro, liquidación y CLV.

El CLV (closing line value) compara la cuota tomada contra la del
cierre del mercado: la métrica que distingue habilidad de suerte en
decenas de apuestas en vez de miles. Cierre = último snapshot de
``odds_snapshots`` anterior al inicio del juego (mismo book si existe;
si no, mediana entre books); null si no hay snapshot pre-juego.

Funciones puras sobre DataFrames: la CLI orquesta la persistencia
(tabla ``bets``, clave ``bet_id``).
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import polars as pl

from mlb_quant.betting.value import normalized_team

logger = logging.getLogger(__name__)

BETS_KEYS = ("bet_id",)

BETS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "bet_id": pl.Utf8,
    "placed_at_utc": pl.Utf8,
    "game_pk": pl.Int64,
    "game_date": pl.Date,
    "market": pl.Utf8,
    "selection": pl.Utf8,
    "team_name": pl.Utf8,
    "book": pl.Utf8,
    "decimal_odds": pl.Float64,
    "stake": pl.Float64,
    "p_model": pl.Float64,
    "status": pl.Utf8,
    "settled_at_utc": pl.Utf8,
    "profit": pl.Float64,
    "closing_odds": pl.Float64,
    "clv_pct": pl.Float64,
}

#: Tolerancia para emparejar el commence del snapshot con el juego.
_MATCH_TOLERANCE_HOURS = 12


def create_bet(
    game: dict[str, Any],
    selection: str,
    decimal_odds: float,
    stake: float,
    book: str,
    p_model: float | None = None,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Construye la fila de una apuesta nueva (status ``open``).

    Args:
        game: Fila de ``games`` (``game_pk``, ``game_date``, nombres).
        selection: ``home`` | ``away``.
        decimal_odds: Cuota decimal tomada (> 1).
        stake: Monto apostado (> 0).
        book: Casa de apuestas.
        p_model: Probabilidad del modelo al apostar (opcional).
        now: Instante inyectable (tests).

    Returns:
        Una fila con esquema :data:`BETS_SCHEMA`.

    Raises:
        ValueError: Selección, cuota o stake inválidos.
    """
    if selection not in ("home", "away"):
        msg = f"selection debe ser home o away, no {selection!r}."
        raise ValueError(msg)
    if decimal_odds <= 1.0:
        msg = f"La cuota decimal debe ser > 1: {decimal_odds}"
        raise ValueError(msg)
    if stake <= 0:
        msg = f"El stake debe ser positivo: {stake}"
        raise ValueError(msg)
    placed = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    team = game["home_team_name"] if selection == "home" else game["away_team_name"]
    row = {
        "bet_id": uuid.uuid4().hex[:12],
        "placed_at_utc": placed,
        "game_pk": game["game_pk"],
        "game_date": game["game_date"],
        "market": "moneyline",
        "selection": selection,
        "team_name": team,
        "book": book,
        "decimal_odds": decimal_odds,
        "stake": stake,
        "p_model": p_model,
        "status": "open",
        "settled_at_utc": None,
        "profit": None,
        "closing_odds": None,
        "clv_pct": None,
    }
    return pl.DataFrame([row], schema=BETS_SCHEMA)


def settle_bets(
    bets: pl.DataFrame, games: pl.DataFrame, now: datetime | None = None
) -> pl.DataFrame:
    """Liquida las apuestas abiertas cuyos juegos ya finalizaron.

    Args:
        bets: Ledger completo (esquema :data:`BETS_SCHEMA`).
        games: Tabla ``games`` (``game_pk``, ``status``, marcadores).
        now: Instante inyectable (tests).

    Returns:
        SOLO las filas recién liquidadas, con ``status``, ``profit`` y
        ``settled_at_utc`` actualizados (listas para upsert). Empates
        (marcador igual: juego suspendido raro) quedan ``void`` con
        profit 0.
    """
    open_bets = bets.filter(pl.col("status") == "open")
    if open_bets.is_empty():
        return open_bets
    finals = games.filter(pl.col("status") == "Final").select(
        "game_pk", "home_score", "away_score"
    )
    joined = open_bets.join(finals, on="game_pk", how="inner")
    if joined.is_empty():
        return joined.drop("home_score", "away_score")

    settled_at = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    home_won = pl.col("home_score") > pl.col("away_score")
    away_won = pl.col("home_score") < pl.col("away_score")
    bet_won = pl.when(pl.col("selection") == "home").then(home_won).otherwise(away_won)
    bet_lost = pl.when(pl.col("selection") == "home").then(away_won).otherwise(home_won)
    settled = joined.with_columns(
        pl.when(bet_won)
        .then(pl.lit("won"))
        .when(bet_lost)
        .then(pl.lit("lost"))
        .otherwise(pl.lit("void"))
        .alias("status"),
        pl.when(bet_won)
        .then(pl.col("stake") * (pl.col("decimal_odds") - 1.0))
        .when(bet_lost)
        .then(-pl.col("stake"))
        .otherwise(0.0)
        .alias("profit"),
        pl.lit(settled_at).alias("settled_at_utc"),
    )
    logger.info("Liquidadas %d apuesta(s).", len(settled))
    return settled.drop("home_score", "away_score")


def closing_odds(
    bets: pl.DataFrame, snapshots: pl.DataFrame, games: pl.DataFrame
) -> pl.DataFrame:
    """Cuota de cierre y CLV por apuesta desde el histórico de snapshots.

    Cierre = último snapshot con ``snapshot_utc`` anterior al commence
    del evento; se prefiere la cuota del mismo book y si no existe se
    usa la mediana entre books en ese snapshot. Emparejado por
    ``event_id == 'manual-{game_pk}'`` primero y por nombres
    normalizados + tolerancia de 12 h como respaldo.

    Args:
        bets: Filas de apuestas (cualquier subconjunto del ledger).
        snapshots: Tabla ``odds_snapshots``.
        games: Tabla ``games`` (nombres y ``game_datetime_utc``).

    Returns:
        ``bet_id``, ``closing_odds``, ``clv_pct`` (nulls sin snapshot
        pre-juego). ``clv_pct = cuota_tomada / cierre - 1``.
    """
    if bets.is_empty() or snapshots.is_empty():
        return pl.DataFrame(
            schema={"bet_id": pl.Utf8, "closing_odds": pl.Float64, "clv_pct": pl.Float64}
        )
    h2h = snapshots.filter(pl.col("market") == "h2h").with_columns(
        normalized_team("home_team").alias("_home_norm"),
        normalized_team("away_team").alias("_away_norm"),
        normalized_team("outcome").alias("_outcome_norm"),
        pl.col("commence_time_utc").str.to_datetime(time_zone="UTC").alias("_commence"),
        pl.col("snapshot_utc").str.to_datetime(time_zone="UTC").alias("_snapshot"),
    )
    game_meta = games.select(
        "game_pk",
        "home_team_name",
        "away_team_name",
        pl.col("game_datetime_utc").str.to_datetime(time_zone="UTC").alias("_game_dt"),
    )

    rows = []
    for bet in bets.join(game_meta, on="game_pk", how="left").iter_rows(named=True):
        closing = _closing_for_bet(bet, h2h)
        clv = bet["decimal_odds"] / closing - 1.0 if closing else None
        rows.append(
            {"bet_id": bet["bet_id"], "closing_odds": closing, "clv_pct": clv}
        )
    return pl.DataFrame(
        rows,
        schema={"bet_id": pl.Utf8, "closing_odds": pl.Float64, "clv_pct": pl.Float64},
    )


def ledger_summary(bets: pl.DataFrame) -> dict[str, float]:
    """Resumen agregado del ledger.

    Args:
        bets: Ledger completo.

    Returns:
        ``n``, ``n_open``, ``staked`` (liquidado), ``profit``, ``roi``,
        ``hit_rate``, ``clv_mean``, ``n_clv``.
    """
    settled = bets.filter(pl.col("status").is_in(["won", "lost", "void"]))
    staked = float(settled["stake"].sum()) if not settled.is_empty() else 0.0
    profit = float(settled["profit"].sum()) if not settled.is_empty() else 0.0
    decided = settled.filter(pl.col("status") != "void")
    hit_rate = (
        cast(float, (decided["status"] == "won").mean())
        if not decided.is_empty()
        else float("nan")
    )
    clv = bets.filter(pl.col("clv_pct").is_not_null())["clv_pct"]
    return {
        "n": float(len(bets)),
        "n_open": float(len(bets.filter(pl.col("status") == "open"))),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else float("nan"),
        "hit_rate": hit_rate,
        "clv_mean": cast(float, clv.mean()) if len(clv) else float("nan"),
        "n_clv": float(len(clv)),
    }


def _closing_for_bet(bet: dict[str, Any], h2h: pl.DataFrame) -> float | None:
    """Cuota de cierre para una apuesta (None sin snapshot pre-juego)."""
    event = h2h.filter(pl.col("event_id") == f"manual-{bet['game_pk']}")
    if event.is_empty() and bet.get("_game_dt") is not None:
        home_norm = (bet.get("home_team_name") or "").lower().strip()
        away_norm = (bet.get("away_team_name") or "").lower().strip()
        event = h2h.filter(
            (pl.col("_home_norm") == home_norm)
            & (pl.col("_away_norm") == away_norm)
            & (
                (pl.col("_commence") - bet["_game_dt"]).abs()
                <= pl.duration(hours=_MATCH_TOLERANCE_HOURS)
            )
        )
    if event.is_empty():
        return None

    team = (
        bet.get("home_team_name")
        if bet["selection"] == "home"
        else bet.get("away_team_name")
    ) or bet["team_name"]
    pre_game = event.filter(
        (pl.col("_outcome_norm") == team.lower().strip())
        & (pl.col("_snapshot") < pl.col("_commence"))
    )
    if pre_game.is_empty():
        return None
    last_snapshot = pre_game.filter(pl.col("_snapshot") == pl.col("_snapshot").max())
    same_book = last_snapshot.filter(pl.col("book") == bet["book"])
    source = same_book if not same_book.is_empty() else last_snapshot
    return cast(float, source["decimal_odds"].median())
