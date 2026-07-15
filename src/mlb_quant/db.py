"""Capa de acceso a DuckDB.

Única responsabilidad: persistencia. Las fuentes de ingesta devuelven
DataFrames; esta capa los escribe de forma idempotente (upsert por claves).
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

type Frame = pl.DataFrame | pd.DataFrame


class Database:
    """Envoltorio fino sobre un archivo DuckDB.

    Attributes:
        path: Ruta del archivo ``.duckdb``.
    """

    def __init__(self, path: Path) -> None:
        """Inicializa la base de datos, creando el directorio si no existe.

        Args:
            path: Ruta del archivo DuckDB.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Abre una conexión que se cierra automáticamente al salir."""
        con = duckdb.connect(str(self.path))
        try:
            yield con
        finally:
            con.close()

    def table_exists(self, table: str) -> bool:
        """Indica si ``table`` existe en el esquema principal."""
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()
        return bool(row and row[0])

    def upsert(self, table: str, df: Frame, keys: Sequence[str]) -> int:
        """Inserta ``df`` en ``table`` reemplazando filas con las mismas claves.

        Idempotente: re-ingerir el mismo rango de fechas no duplica filas.
        Si la tabla no existe, se crea con el esquema del DataFrame.

        Args:
            table: Nombre de la tabla destino.
            df: Datos a escribir (Polars o Pandas).
            keys: Columnas que identifican una fila de forma única.

        Returns:
            Número de filas escritas.

        Raises:
            ValueError: Si ``df`` no contiene alguna columna clave.
        """
        n_rows = len(df)
        if n_rows == 0:
            logger.info("upsert en %s: DataFrame vacío, nada que hacer.", table)
            return 0
        missing = [k for k in keys if k not in df.columns]
        if missing:
            msg = f"Claves ausentes en el DataFrame: {missing}"
            raise ValueError(msg)

        with self.connect() as con:
            con.register("_incoming", df)
            if not self._table_exists_on(con, table):
                con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _incoming')
                logger.info("Tabla %s creada con %d filas.", table, n_rows)
                return n_rows

            self._add_missing_columns(con, table)
            predicate = " AND ".join(f't."{k}" = d."{k}"' for k in keys)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f'DELETE FROM "{table}" t '
                    f"WHERE EXISTS (SELECT 1 FROM _incoming d WHERE {predicate})"
                )
                con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM _incoming')
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        logger.info("upsert en %s: %d filas escritas.", table, n_rows)
        return n_rows

    def query(self, sql: str) -> pl.DataFrame:
        """Ejecuta ``sql`` y devuelve el resultado como Polars DataFrame."""
        with self.connect() as con:
            return con.execute(sql).pl()

    @staticmethod
    def _add_missing_columns(con: duckdb.DuckDBPyConnection, table: str) -> None:
        """Evoluciona el esquema: añade a ``table`` columnas nuevas de ``_incoming``.

        Las columnas que la tabla tiene y el DataFrame no, las rellena
        ``INSERT BY NAME`` con NULL; el caso inverso requiere ALTER TABLE.
        """
        existing = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [table],
            ).fetchall()
        }
        incoming = con.execute("DESCRIBE _incoming").fetchall()
        for column_name, column_type, *_ in incoming:
            if column_name not in existing:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column_name}" {column_type}')
                logger.info("Tabla %s: columna nueva %s (%s).", table, column_name, column_type)

    @staticmethod
    def _table_exists_on(con: duckdb.DuckDBPyConnection, table: str) -> bool:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return bool(row and row[0])
