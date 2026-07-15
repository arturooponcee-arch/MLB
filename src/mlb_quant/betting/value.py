"""Detección de apuestas con valor: probabilidades del modelo vs. mercado.

Una apuesta tiene valor si ``EV = p * cuota - 1`` supera el umbral,
evaluada contra el mejor precio disponible entre casas. El sizing lo da
Kelly fraccional (:mod:`mlb_quant.bankroll.kelly`).

Solo moneyline por ahora: los totals del mercado traen una línea
distinta por casa y el simulador evalúa una línea fija por juego; se
cruzarán cuando el simulador acepte la línea como parámetro por juego.
"""

import logging

import polars as pl

from mlb_quant.bankroll.kelly import recommend_stake

logger = logging.getLogger(__name__)

# Tolerancia para emparejar evento de cuotas con juego programado.
# Cubre cambios de horario; evita cruzar juegos de días distintos de
# una misma serie (mismo local y visitante).
_MATCH_TOLERANCE_HOURS = 12


def latest_snapshot(odds: pl.DataFrame) -> pl.DataFrame:
    """Filtra ``odds`` al snapshot más reciente (cuotas vigentes)."""
    if odds.is_empty():
        return odds
    return odds.filter(pl.col("snapshot_utc") == pl.col("snapshot_utc").max())


def best_moneyline_prices(odds: pl.DataFrame) -> pl.DataFrame:
    """Mejor cuota moneyline por evento y equipo, entre todas las casas.

    Usa solo el snapshot más reciente: las cuotas viejas ya no se pueden
    tomar.

    Args:
        odds: Filas con esquema :data:`mlb_quant.ingestion.odds_api.ODDS_SCHEMA`.

    Returns:
        Una fila por evento x equipo: ``event_id``, ``commence_time_utc``,
        ``home_team``, ``away_team``, ``outcome``, ``decimal_odds``, ``book``.
    """
    if odds.is_empty():
        return odds
    h2h = latest_snapshot(odds).filter(pl.col("market") == "h2h")
    if h2h.is_empty():
        return h2h
    return (
        h2h.sort("decimal_odds", descending=True)
        .unique(subset=["event_id", "outcome"], keep="first", maintain_order=True)
        .select(
            "event_id",
            "commence_time_utc",
            "home_team",
            "away_team",
            "outcome",
            "decimal_odds",
            "book",
        )
    )


def find_moneyline_value(
    games: pl.DataFrame,
    odds: pl.DataFrame,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.5,
    max_fraction: float = 0.05,
    min_ev: float = 0.0,
) -> pl.DataFrame:
    """Cruza predicciones con cuotas y devuelve las apuestas con valor.

    Evalúa ambos lados (local y visitante) de cada juego contra el mejor
    precio del mercado y calcula EV y stake Kelly fraccional.

    Args:
        games: Salida de ``predict_upcoming_games`` (``game_pk``,
            ``game_datetime_utc``, nombres de equipos y ``p_home_win``).
        odds: Snapshot(s) de cuotas con esquema ``ODDS_SCHEMA``.
        bankroll: Bankroll actual para el sizing.
        kelly_multiplier: Fracción de Kelly (0.5 = half).
        max_fraction: Tope duro como fracción del bankroll.
        min_ev: EV mínimo por unidad para reportar la apuesta.

    Returns:
        Una fila por apuesta con valor, ordenada por EV descendente:
        equipos, lado, ``p_model``, ``fair_odds``, ``best_odds``, ``book``,
        ``ev``, ``kelly_stake``, ``capped``. Vacío si nada supera ``min_ev``.
    """
    best = best_moneyline_prices(odds)
    if games.is_empty() or best.is_empty():
        return pl.DataFrame()

    matched = _match_events(games, best)
    unmatched = len(games) - len(matched)
    if unmatched:
        logger.warning(
            "%d juego(s) sin evento de cuotas emparejado (nombres u horarios).", unmatched
        )
    if matched.is_empty():
        return pl.DataFrame()

    sides = matched.select(
        "game_pk",
        "event_id",
        "game_datetime_utc",
        "away_team_name",
        "home_team_name",
        "p_home_win",
    )
    long = pl.concat(
        [
            sides.with_columns(
                pl.lit("home").alias("side"),
                pl.col("home_team_name").alias("team"),
                pl.col("p_home_win").alias("p_model"),
            ),
            sides.with_columns(
                pl.lit("away").alias("side"),
                pl.col("away_team_name").alias("team"),
                (1.0 - pl.col("p_home_win")).alias("p_model"),
            ),
        ]
    ).drop("p_home_win")

    priced = long.with_columns(_normalized("team").alias("outcome_norm")).join(
        best.with_columns(_normalized("outcome").alias("outcome_norm")).select(
            "event_id", "outcome_norm", "decimal_odds", "book"
        ),
        on=["event_id", "outcome_norm"],
        how="inner",
    )

    rows = []
    for row in priced.iter_rows(named=True):
        if not 0.0 < row["p_model"] < 1.0:
            continue
        rec = recommend_stake(
            bankroll,
            row["p_model"],
            row["decimal_odds"],
            kelly_multiplier=kelly_multiplier,
            max_fraction=max_fraction,
        )
        rows.append(
            {
                "game_pk": row["game_pk"],
                "game_datetime_utc": row["game_datetime_utc"],
                "away_team_name": row["away_team_name"],
                "home_team_name": row["home_team_name"],
                "side": row["side"],
                "team": row["team"],
                "p_model": row["p_model"],
                "fair_odds": rec.fair_odds,
                "best_odds": row["decimal_odds"],
                "book": row["book"],
                "ev": rec.ev,
                "kelly_stake": rec.stake,
                "capped": rec.capped,
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).filter(pl.col("ev") > min_ev).sort("ev", descending=True)


def _match_events(games: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Empareja juegos programados con eventos de cuotas.

    Cruza por nombres normalizados de local y visitante y desempata por
    cercanía horaria (una serie repite el mismo par varios días; los
    doubleheaders lo repiten el mismo día).
    """
    events = (
        best.select("event_id", "commence_time_utc", "home_team", "away_team")
        .unique(subset=["event_id"])
        .with_columns(
            _normalized("home_team").alias("home_norm"),
            _normalized("away_team").alias("away_norm"),
            pl.col("commence_time_utc").str.to_datetime(time_zone="UTC").alias("event_dt"),
        )
    )
    candidates = (
        games.select(
            "game_pk", "game_datetime_utc", "home_team_name", "away_team_name", "p_home_win"
        )
        .with_columns(
            _normalized("home_team_name").alias("home_norm"),
            _normalized("away_team_name").alias("away_norm"),
            pl.col("game_datetime_utc").str.to_datetime(time_zone="UTC").alias("game_dt"),
        )
        .join(events, on=["home_norm", "away_norm"], how="inner")
        .with_columns((pl.col("game_dt") - pl.col("event_dt")).abs().alias("dt_diff"))
        .filter(pl.col("dt_diff") <= pl.duration(hours=_MATCH_TOLERANCE_HOURS))
    )
    return (
        candidates.sort("dt_diff")
        .unique(subset=["game_pk"], keep="first", maintain_order=True)
        .unique(subset=["event_id"], keep="first", maintain_order=True)
    )


def _normalized(column: str) -> pl.Expr:
    """Nombre de equipo normalizado para cruzar fuentes distintas."""
    return pl.col(column).str.to_lowercase().str.strip_chars()
