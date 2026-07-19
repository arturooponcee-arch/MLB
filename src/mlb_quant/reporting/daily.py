"""Pipeline de predicción del día y reporte diario.

Centraliza el flujo features-futuras -> modelos -> simulación que
comparten el dashboard y los reportes exportables.
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl

from mlb_quant.analysts.brain import DailyParlays, generate_daily_parlays
from mlb_quant.db import Database
from mlb_quant.feature_engineering.upcoming import build_upcoming_features
from mlb_quant.models.ensemble import CalibratedEnsembleWinModel
from mlb_quant.models.poisson import PoissonRunsModel
from mlb_quant.models.props import predict_upcoming_strikeouts
from mlb_quant.reporting.exports import DEFAULT_FORMATS, ExportFormat, export_frame, to_markdown
from mlb_quant.simulations.monte_carlo import SimulationConfig, simulate_games

logger = logging.getLogger(__name__)


def predict_upcoming_games(db: Database, game_date: date, sims: int) -> pl.DataFrame:
    """Predice los juegos programados de una fecha (ML calibrado + Monte Carlo).

    Args:
        db: Warehouse con ``features_game`` y calendario del día.
        game_date: Fecha de los juegos.
        sims: Simulaciones por juego.

    Returns:
        Una fila por juego: metadatos + ``p_home_win`` + columnas ``sim_*``.
        Vacío si no hay juegos programados.
    """
    upcoming = build_upcoming_features(db, game_date, game_date)
    if upcoming.is_empty():
        return upcoming
    training = db.query("SELECT * FROM features_game")

    ensemble = CalibratedEnsembleWinModel().fit(training)
    poisson = PoissonRunsModel().fit(training)

    win = ensemble.predict_home_win(upcoming)
    lambdas = poisson.predict_lambdas(upcoming)
    sims_df = simulate_games(lambdas, SimulationConfig(n_sims=sims))

    # Columnas de contexto para los analistas (clima, calidad de abridores);
    # ausentes si el bloque correspondiente se omitió en el builder.
    context_cols = [
        c
        for c in (
            "wx_temperature_c",
            "wx_wind_out",
            "wx_roof_possible",
            "home_spq_xera",
            "away_spq_xera",
        )
        if c in upcoming.columns
    ]
    return (
        upcoming.select(
            "game_pk",
            "game_datetime_utc",
            "home_team_name",
            "away_team_name",
            "home_probable_pitcher_name",
            "away_probable_pitcher_name",
            *context_cols,
        )
        .join(win, on="game_pk")
        .join(sims_df, on="game_pk")
        .sort("game_datetime_utc")
    )


def generate_daily_report(
    db: Database,
    game_date: date,
    output_dir: Path,
    sims: int = 10_000,
    k_line: float = 5.5,
    formats: tuple[ExportFormat, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Genera el reporte diario: juegos + props de K, en MD/CSV/Excel.

    Args:
        db: Warehouse completo.
        game_date: Fecha de los juegos.
        output_dir: Directorio destino (``reports/daily``).
        sims: Simulaciones Monte Carlo por juego.
        k_line: Línea del prop de strikeouts.
        formats: Formatos a exportar por tabla.

    Returns:
        Rutas escritas (tablas exportadas + resumen Markdown).
    """
    games = predict_upcoming_games(db, game_date, sims)
    props = predict_upcoming_strikeouts(db, game_date, line=k_line)
    parlays = generate_daily_parlays(games, props)

    written: list[Path] = []
    stamp = game_date.isoformat()
    if not games.is_empty():
        games_out = games.select(
            "away_team_name",
            "home_team_name",
            "away_probable_pitcher_name",
            "home_probable_pitcher_name",
            (1 - pl.col("p_home_win")).round(3).alias("p_away"),
            pl.col("p_home_win").round(3).alias("p_home"),
            (1 / pl.col("p_home_win")).round(2).alias("cuota_justa_home"),
            (1 / (1 - pl.col("p_home_win"))).round(2).alias("cuota_justa_away"),
            pl.col("sim_exp_total").round(1).alias("total_esperado"),
            pl.col("sim_p_over").round(3).alias("p_over_8_5"),
            pl.col("sim_p_home_runline").round(3).alias("p_home_rl_1_5"),
            pl.col("sim_p_f5_home_ml").round(3).alias("p_f5_home"),
        )
        written += export_frame(games_out, output_dir / f"{stamp}_juegos", formats)
    if not props.is_empty():
        props_out = props.select(
            "pitcher_name",
            "opponent_name",
            pl.col("lambda").round(2).alias("k_esperados"),
            "line",
            pl.col("p_over").round(3),
            (1 / pl.col("p_over")).round(2).alias("cuota_justa_over"),
            (1 / (1 - pl.col("p_over"))).round(2).alias("cuota_justa_under"),
        ).sort("p_over", descending=True)
        written += export_frame(props_out, output_dir / f"{stamp}_props_k", formats)
    parlays_out = parlays.to_frame()
    if not parlays_out.is_empty():
        written += export_frame(parlays_out, output_dir / f"{stamp}_combinadas", formats)

    summary = output_dir / f"{stamp}_resumen.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        _summary_markdown(game_date, games, props, k_line, parlays), encoding="utf-8"
    )
    written.append(summary)
    logger.info("Reporte diario %s: %d archivos.", stamp, len(written))
    return written


def _summary_markdown(
    game_date: date,
    games: pl.DataFrame,
    props: pl.DataFrame,
    k_line: float,
    parlays: DailyParlays | None = None,
) -> str:
    lines = [f"# Reporte diario MLB Quant — {game_date}", ""]
    if games.is_empty():
        lines.append("_Sin juegos programados._")
        return "\n".join(lines) + "\n"
    lines += [
        f"**{len(games)} juegos.** Moneyline: ensemble calibrado; "
        "resto de mercados: Monte Carlo 10k sims.",
        "",
        "## Juegos",
        "",
        to_markdown(
            games.select(
                pl.col("away_team_name").alias("visita"),
                pl.col("home_team_name").alias("local"),
                pl.col("p_home_win").round(3).alias("p_local"),
                (1 / pl.col("p_home_win")).round(2).alias("justa_local"),
                pl.col("sim_exp_total").round(1).alias("total_esp"),
                pl.col("sim_p_over").round(3).alias("p_over_8.5"),
            )
        ),
    ]
    if not props.is_empty():
        lines += [
            f"## Strikeouts del abridor (línea {k_line})",
            "",
            to_markdown(
                props.select(
                    pl.col("pitcher_name").alias("pitcher"),
                    pl.col("opponent_name").alias("rival"),
                    pl.col("lambda").round(2).alias("k_esp"),
                    pl.col("p_over").round(3),
                ).sort("p_over", descending=True)
            ),
        ]
    if parlays is not None and parlays.parlays:
        lines += _parlays_markdown(parlays)
    lines.append(
        "\n_Cuota justa = 1/p: una apuesta manual solo tiene valor si el book paga más._"
    )
    return "\n".join(lines) + "\n"


def _parlays_markdown(parlays: DailyParlays) -> list[str]:
    """Sección de combinadas sugeridas por el cerebro pronosticador."""
    lines = ["", "## Combinadas del día", ""]
    for parlay in parlays.parlays:
        lines.append(
            f"**{parlay.name.capitalize()}** — p combinada "
            f"{parlay.p_combined:.3f}, cuota justa {parlay.fair_odds:.2f}"
        )
        for leg in parlay.legs.iter_rows(named=True):
            lines.append(f"- {leg['selection']} (p {leg['p']:.3f})")
        if parlay.substitute is not None:
            sub = parlay.substitute.row(0, named=True)
            lines.append(
                f"- _Suplente si el prop no está en tu book: "
                f"{sub['selection']} (p {sub['p']:.3f})_"
            )
        lines.append("")
    if not parlays.notes.is_empty():
        lines += ["### Notas de analistas", ""]
        for note in parlays.notes.iter_rows(named=True):
            where = f" — {note['matchup']}" if note.get("matchup") else ""
            lines.append(f"- [{note['analyst']}] {note['note']}{where}")
        lines.append("")
    lines.append(
        "_Una pierna por juego (evita correlación). La p combinada asume "
        "independencia entre juegos; vale solo si el book paga más que la cuota justa._"
    )
    return lines
