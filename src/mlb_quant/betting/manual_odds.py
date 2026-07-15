"""Cuotas manuales desde CSV: mismo esquema que un snapshot de API.

Camino sin API de cuotas: el usuario teclea las cuotas de su book en un
CSV (plantilla con ``mlb bets template``) y el sistema las convierte al
esquema de ``odds_snapshots`` y las persiste igual que un snapshot de
The Odds API. El warehouse acumula así el histórico del mercado aunque
la fuente sea manual: sirve para medir CLV y para contrastar el modelo
contra el mercado con el tiempo.

Formato del CSV (una fila por juego; ``book`` opcional, default
``manual``; varias casas = repetir la fila con otro ``book``)::

    away_team,home_team,odds_away,odds_home,book
    New York Yankees,Boston Red Sox,2.05,1.85,caliente
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from mlb_quant.ingestion.odds_api import ODDS_SCHEMA

logger = logging.getLogger(__name__)

MANUAL_CSV_COLUMNS = ("away_team", "home_team", "odds_away", "odds_home")
DEFAULT_BOOK = "manual"


def load_manual_odds(
    path: Path,
    games: pl.DataFrame,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Lee un CSV de cuotas manuales y lo convierte al esquema de snapshots.

    Cruza cada fila con los juegos programados por nombres normalizados
    de equipos. Los errores de captura (equipo mal escrito, cuota <= 1,
    fila repetida) cortan con mensaje explícito: el CSV se teclea a mano
    y un cruce silenciosamente incompleto escondería el error.

    Args:
        path: Ruta del CSV (columnas :data:`MANUAL_CSV_COLUMNS` + ``book``
            opcional).
        games: Juegos de la fecha (``game_pk``, ``game_datetime_utc``,
            ``home_team_name``, ``away_team_name``); típicamente la salida
            de ``predict_upcoming_games``.
        now: Instante "ahora" inyectable (para tests del snapshot).

    Returns:
        Filas con esquema :data:`~mlb_quant.ingestion.odds_api.ODDS_SCHEMA`
        (mercado ``h2h``, un ``event_id`` sintético ``manual-{game_pk}``).
        En un doubleheader (mismo par de equipos dos veces el mismo día),
        la fila del CSV aplica a ambos juegos.

    Raises:
        ValueError: Columnas ausentes, cuotas fuera de rango, filas
            duplicadas o equipos que no cruzan con los juegos del día.
    """
    raw = pl.read_csv(path)
    missing = [c for c in MANUAL_CSV_COLUMNS if c not in raw.columns]
    if missing:
        msg = f"Columnas ausentes en {path.name}: {missing}. Genera una con: mlb bets template"
        raise ValueError(msg)
    if "book" not in raw.columns:
        raw = raw.with_columns(pl.lit(DEFAULT_BOOK).alias("book"))
    raw = raw.with_columns(
        pl.col("book").cast(pl.Utf8).fill_null(DEFAULT_BOOK),
        pl.col("odds_away").cast(pl.Float64, strict=False),
        pl.col("odds_home").cast(pl.Float64, strict=False),
        _normalized("away_team").alias("away_norm"),
        _normalized("home_team").alias("home_norm"),
    )

    bad_odds = raw.filter(
        pl.col("odds_away").is_null()
        | pl.col("odds_home").is_null()
        | (pl.col("odds_away") <= 1.0)
        | (pl.col("odds_home") <= 1.0)
    )
    if not bad_odds.is_empty():
        pairs = [f"{r['away_team']} @ {r['home_team']}" for r in bad_odds.iter_rows(named=True)]
        msg = f"Cuotas vacías o <= 1 (decimales, > 1) en: {pairs}"
        raise ValueError(msg)

    duplicate_keys = ["away_norm", "home_norm", "book"]
    if raw.select(duplicate_keys).n_unique() != len(raw):
        msg = "Filas repetidas (mismo juego y book). Deja una cuota por juego y book."
        raise ValueError(msg)

    matched = raw.join(
        games.select(
            "game_pk",
            "game_datetime_utc",
            "home_team_name",
            "away_team_name",
        ).with_columns(
            _normalized("home_team_name").alias("home_norm"),
            _normalized("away_team_name").alias("away_norm"),
        ),
        on=["away_norm", "home_norm"],
        how="left",
    )
    unmatched = matched.filter(pl.col("game_pk").is_null())
    if not unmatched.is_empty():
        pairs = [f"{r['away_team']} @ {r['home_team']}" for r in unmatched.iter_rows(named=True)]
        names = sorted(games["away_team_name"].to_list() + games["home_team_name"].to_list())
        msg = (
            f"Sin juego programado que cruce con: {pairs}. "
            f"Usa los nombres oficiales completos: {names}"
        )
        raise ValueError(msg)

    snapshot = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        {
            "snapshot_utc": snapshot,
            "event_id": f"manual-{row['game_pk']}",
            "commence_time_utc": row["game_datetime_utc"],
            "home_team": row["home_team_name"],
            "away_team": row["away_team_name"],
            "book": row["book"],
            "market": "h2h",
            "outcome": outcome,
            "point": None,
            "decimal_odds": price,
            "last_update_utc": snapshot,
        }
        for row in matched.iter_rows(named=True)
        for outcome, price in (
            (row["home_team_name"], row["odds_home"]),
            (row["away_team_name"], row["odds_away"]),
        )
    ]
    logger.info(
        "Cuotas manuales %s: %d fila(s) de CSV -> %d cuotas.", path.name, len(raw), len(rows)
    )
    return pl.DataFrame(rows, schema=ODDS_SCHEMA)


def write_template(path: Path, games: pl.DataFrame) -> int:
    """Escribe una plantilla CSV con los juegos del día y cuotas vacías.

    Args:
        path: Destino del CSV.
        games: Juegos de la fecha (``away_team_name``, ``home_team_name``).

    Returns:
        Número de juegos en la plantilla.
    """
    template = games.select(
        pl.col("away_team_name").alias("away_team"),
        pl.col("home_team_name").alias("home_team"),
        pl.lit(None, dtype=pl.Float64).alias("odds_away"),
        pl.lit(None, dtype=pl.Float64).alias("odds_home"),
        pl.lit(DEFAULT_BOOK).alias("book"),
    ).unique(maintain_order=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    template.write_csv(path)
    return len(template)


def _normalized(column: str) -> pl.Expr:
    """Nombre de equipo normalizado (idéntico a betting.value)."""
    return pl.col(column).str.to_lowercase().str.strip_chars()
