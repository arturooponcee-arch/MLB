"""Contrato y registro de bloques de features.

Un bloque es una unidad independiente que lee tablas del warehouse y
devuelve un DataFrame de features. El grano define cómo lo une el builder:

- ``game``: una fila por ``game_pk`` (park, clima).
- ``team_game``: una fila por ``(game_pk, team_id)`` (forma del equipo,
  pitcher abridor); el builder lo une dos veces con prefijos
  ``home_`` / ``away_``.

Regla de oro anti-leakage: toda feature debe calcularse SOLO con
información disponible antes del primer lanzamiento del juego.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import polars as pl

from mlb_quant.db import Database

type Grain = Literal["game", "team_game"]

#: Columnas clave por grano; el resto son features.
GRAIN_KEYS: dict[str, tuple[str, ...]] = {
    "game": ("game_pk",),
    "team_game": ("game_pk", "team_id"),
}


class FeatureBlock(ABC):
    """Base de todo bloque de features.

    Attributes:
        name: Identificador único del bloque.
        grain: Grano de la salida (``game`` o ``team_game``).
        requires: Tablas del warehouse que el bloque necesita.
    """

    name: ClassVar[str]
    grain: ClassVar[Grain]
    requires: ClassVar[tuple[str, ...]]

    @abstractmethod
    def build(self, db: Database) -> pl.DataFrame:
        """Calcula las features del bloque para todo el warehouse.

        Args:
            db: Base de datos con las tablas de ingesta.

        Returns:
            DataFrame con las columnas clave del grano más las features.
        """

    def check_requirements(self, db: Database) -> list[str]:
        """Devuelve las tablas requeridas que faltan en el warehouse."""
        return [t for t in self.requires if not db.table_exists(t)]


_REGISTRY: dict[str, type[FeatureBlock]] = {}


def register(cls: type[FeatureBlock]) -> type[FeatureBlock]:
    """Decorador: registra un bloque por su ``name``.

    Raises:
        ValueError: Si ya existe un bloque con el mismo nombre.
    """
    if cls.name in _REGISTRY:
        msg = f"Bloque duplicado: {cls.name!r}"
        raise ValueError(msg)
    _REGISTRY[cls.name] = cls
    return cls


def get_blocks(names: list[str] | None = None) -> list[FeatureBlock]:
    """Instancia bloques registrados.

    Args:
        names: Bloques a instanciar; ``None`` = todos.

    Returns:
        Lista de instancias en orden de registro.

    Raises:
        ValueError: Si se pide un bloque no registrado.
    """
    if names is None:
        return [cls() for cls in _REGISTRY.values()]
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        msg = f"Bloques no registrados: {unknown}. Disponibles: {sorted(_REGISTRY)}"
        raise ValueError(msg)
    return [_REGISTRY[n]() for n in names]
