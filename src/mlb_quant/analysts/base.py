"""Tipos compartidos del equipo de analistas.

Arquitectura "cerebro + subagentes" determinística: cada analista es un
módulo puro (sin LLM) que lee el estado del día y emite señales tipadas;
el cerebro (:mod:`mlb_quant.analysts.brain`) las consolida en combinadas.

Contrato: cada analista recibe el :class:`DailyContext` y devuelve un
:class:`AnalystReport` con señales por ``game_pk`` y notas legibles. Los
analistas no se hablan entre sí — solo el cerebro ve todo.
"""

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class DailyContext:
    """Estado del día que comparten todos los analistas.

    Attributes:
        games: Salida de ``predict_upcoming_games`` — una fila por juego
            con metadatos, ``p_home_win``, columnas ``sim_*`` y (si el
            builder las produjo) columnas de clima ``wx_*`` y de calidad
            de abridores ``*_spq_*`` / ``*_sp_days_rest``.
        props: Salida de ``predict_upcoming_strikeouts`` (puede ser vacía).
    """

    games: pl.DataFrame
    props: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass(frozen=True)
class AnalystReport:
    """Resultado de un analista.

    Attributes:
        analyst: Nombre corto del analista (``clima``, ``jugadores``...).
        signals: Una fila por ``game_pk`` con las señales numéricas del
            analista. Vacío si no hay datos.
        notes: Filas ``(game_pk, note)`` con hallazgos legibles para el
            reporte. Vacío si no hay nada que destacar.
    """

    analyst: str
    signals: pl.DataFrame
    notes: pl.DataFrame = field(default_factory=pl.DataFrame)


def has_columns(df: pl.DataFrame, *names: str) -> bool:
    """True si ``df`` contiene todas las columnas ``names``."""
    return all(name in df.columns for name in names)


def empty_notes() -> pl.DataFrame:
    """Frame de notas vacío con el esquema canónico."""
    return pl.DataFrame(schema={"game_pk": pl.Int64, "note": pl.String})
