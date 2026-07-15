"""Modelos de props por jugador: conteos Poisson por mercado.

Un modelo genérico (:class:`PropModel`) entrenado sobre frames
específicos: strikeouts del abridor (por apertura) y stats del bateador
(hits, bases totales, HR, walks por juego). La probabilidad de over/under
sale de la PMF Poisson (reutiliza la malla de ``markets``).
"""

import logging

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_quant.db import Database
from mlb_quant.feature_engineering.rolling import RollingSpec, add_lagged_rolling
from mlb_quant.models.markets import poisson_pmf_matrix

logger = logging.getLogger(__name__)

#: Columnas que nunca son features en frames de props.
PROP_META: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "season",
    "player_id",
    "team_id",
    "target",
)

#: Stats de bateador soportadas como target.
BATTER_STATS: tuple[str, ...] = ("hits", "total_bases", "home_runs", "walks")

_PITCHER_K_TARGET_SQL = """
SELECT p.game_pk, p.team_id, p.player_id, p.strike_outs AS target
FROM pitching_lines p
JOIN games g USING (game_pk)
WHERE p.is_starter AND g.status = 'Final' AND g.game_type = 'R'
"""

_BATTER_BASE_SQL = """
SELECT b.game_pk, b.player_id, b.game_date, g.season,
       l.team_id, l.batting_order,
       b.pa, b.hits, b.total_bases, b.home_runs, b.walks
FROM batter_games b
JOIN games g USING (game_pk)
JOIN lineups l ON l.game_pk = b.game_pk AND l.player_id = b.player_id
WHERE g.status = 'Final' AND g.game_type = 'R' AND l.is_starting_lineup
"""

_SAVANT_BATTER_SQL = """
SELECT player_id, year,
       xwoba AS bat_xwoba, k_percent AS bat_k_pct, bb_percent AS bat_bb_pct,
       barrel_batted_rate AS bat_barrel_pct, hard_hit_percent AS bat_hard_hit_pct
FROM savant_statcast_metrics
WHERE player_type = 'batter' AND pa >= 100
"""


class PropModel:
    """Regresión Poisson de un conteo por jugador-juego."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Inicializa el modelo sin entrenar.

        Args:
            alpha: Regularización L2.
        """
        self._columns: list[str] = []
        self._pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000)),
            ]
        )

    def fit(self, frame: pl.DataFrame) -> "PropModel":
        """Entrena sobre un frame con columna ``target``.

        Args:
            frame: Frame de props (meta + features + target).

        Returns:
            El propio modelo.

        Raises:
            ValueError: Si falta la columna target.
        """
        if "target" not in frame.columns:
            msg = "El frame debe incluir la columna 'target'."
            raise ValueError(msg)
        self._columns = [
            c
            for c in frame.columns
            if c not in PROP_META and frame.schema[c].is_numeric()
        ]
        x = frame.select(self._columns).to_pandas()
        self._pipeline.fit(x, frame["target"].to_numpy())
        logger.info(
            "PropModel entrenado: %d filas, %d features.", len(frame), len(self._columns)
        )
        return self

    def predict_lambda(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Predice la tasa esperada del conteo por fila.

        Args:
            frame: Frame con las mismas features (faltantes -> null).

        Returns:
            ``game_pk``, ``player_id``, ``lambda``.
        """
        data = frame.to_pandas()
        for missing in (c for c in self._columns if c not in data.columns):
            data[missing] = float("nan")
        lam = np.clip(
            np.asarray(self._pipeline.predict(data[self._columns]), dtype=np.float64),
            0.01,
            None,
        )
        return pl.DataFrame(
            {
                "game_pk": frame["game_pk"],
                "player_id": frame["player_id"],
                "lambda": lam,
            }
        )

    @staticmethod
    def probability_over(lambdas: np.ndarray, line: float) -> np.ndarray:
        """P(conteo > línea) bajo Poisson.

        Args:
            lambdas: Tasas por fila.
            line: Línea del prop (ej. 5.5).

        Returns:
            Probabilidades de over.
        """
        pmf = poisson_pmf_matrix(lambdas, max_runs=40)
        threshold = int(np.floor(line))
        return 1.0 - pmf[:, : threshold + 1].sum(axis=1)


def build_pitcher_k_frame(db: Database) -> pl.DataFrame:
    """Frame de entrenamiento de strikeouts del abridor.

    Una fila por apertura: forma y nivel del abridor, calidad de contacto
    del lineup RIVAL, parque; target = K del juego. Todo desde
    ``features_game`` (features ya anti-leakage).

    Args:
        db: Warehouse con ``features_game`` y ``pitching_lines``.

    Returns:
        Frame listo para :class:`PropModel`.
    """
    features = db.query("SELECT * FROM features_game")
    targets = db.query(_PITCHER_K_TARGET_SQL)

    park_cols = [c for c in features.columns if c.startswith("park_")]
    sides = []
    for side, opponent in (("home", "away"), ("away", "home")):
        own = [c for c in features.columns if c.startswith((f"{side}_sp_", f"{side}_spq_"))]
        opp = [c for c in features.columns if c.startswith(f"{opponent}_lineup_")]
        renamed = features.select(
            "game_pk",
            "game_date",
            "season",
            pl.col(f"{side}_team_id").alias("team_id"),
            *[pl.col(c).alias(c.removeprefix(f"{side}_")) for c in own],
            *[pl.col(c).alias("opp_" + c.removeprefix(f"{opponent}_")) for c in opp],
            *park_cols,
        )
        sides.append(renamed)
    frame = pl.concat(sides, how="vertical_relaxed")
    return frame.join(targets, on=["game_pk", "team_id"], how="inner").unique(
        subset=["game_pk", "team_id"], keep="first", maintain_order=True
    )


def build_batter_frame(db: Database, stat: str) -> pl.DataFrame:
    """Frame de entrenamiento de un prop de bateador.

    Una fila por bateador titular y juego: forma rolling (15 juegos,
    lagged), calidad Savant de temporada previa, orden al bate, calidad
    del abridor RIVAL y parque; target = la stat del juego.

    Args:
        db: Warehouse con ``batter_games``, ``lineups`` y Savant.
        stat: Una de :data:`BATTER_STATS`.

    Returns:
        Frame listo para :class:`PropModel`.

    Raises:
        ValueError: Si la stat no está soportada.
    """
    if stat not in BATTER_STATS:
        msg = f"Stat no soportada: {stat!r}. Opciones: {BATTER_STATS}"
        raise ValueError(msg)

    base = db.query(_BATTER_BASE_SQL)
    specs = [
        RollingSpec(c, 15, "mean", f"bat_{c}_roll15")
        for c in ("pa", "hits", "total_bases", "home_runs", "walks")
    ]
    base = add_lagged_rolling(
        base, group_by="player_id", sort_by=["game_date", "game_pk"], specs=specs
    )

    savant = db.query(_SAVANT_BATTER_SQL).with_columns(
        (pl.col("year") + 1).alias("_target_season")
    )
    base = base.join(
        savant,
        left_on=["player_id", "season"],
        right_on=["player_id", "_target_season"],
        how="left",
    ).drop("year")

    opponent_sp = _opponent_starter_quality(db)
    base = base.join(opponent_sp, on=["game_pk", "team_id"], how="left")

    park = db.query(
        "SELECT g.game_pk, v.elevation AS park_elevation "
        "FROM games g LEFT JOIN venues v "
        "ON g.venue_id = v.venue_id AND v.season = g.season"
    )
    base = base.join(park, on="game_pk", how="left")

    return base.with_columns(pl.col(stat).alias("target")).drop(
        "pa", "hits", "total_bases", "home_runs", "walks"
    )


def _opponent_starter_quality(db: Database) -> pl.DataFrame:
    """Calidad (temporada previa) del abridor RIVAL, por (game_pk, team_id)."""
    starters = db.query(
        """
        SELECT p.game_pk, p.team_id, p.player_id, g.season, p.outs_recorded
        FROM pitching_lines p JOIN games g USING (game_pk)
        WHERE p.is_starter
        """
    )
    starters = starters.sort("outs_recorded", descending=True).unique(
        subset=["game_pk", "team_id"], keep="first", maintain_order=True
    )
    savant = db.query(
        """
        SELECT s.player_id, s.year,
               s.xwoba AS opp_sp_xwoba, s.k_percent AS opp_sp_k_pct,
               s.whiff_percent AS opp_sp_whiff_pct
        FROM savant_statcast_metrics s
        WHERE s.player_type = 'pitcher' AND s.pa >= 100
        """
    ).with_columns((pl.col("year") + 1).alias("_target_season"))
    quality = starters.join(
        savant,
        left_on=["player_id", "season"],
        right_on=["player_id", "_target_season"],
        how="left",
    )
    # El rival del equipo X en el juego: la otra fila del mismo game_pk.
    games = db.query("SELECT game_pk, home_team_id, away_team_id FROM games")
    quality = quality.join(games, on="game_pk", how="left").with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.col("away_team_id"))
        .otherwise(pl.col("home_team_id"))
        .alias("opponent_team_id")
    )
    return quality.select(
        "game_pk",
        pl.col("opponent_team_id").alias("team_id"),
        "opp_sp_xwoba",
        "opp_sp_k_pct",
        "opp_sp_whiff_pct",
    )
