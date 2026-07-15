"""Fuente Lahman Database vía el paquete R de CRAN.

Histórico profundo (1871-2025) a nivel temporada: base para park factors,
priors bayesianos y contexto de largo plazo.

Por qué CRAN y no pybaseball: el repo GitHub ``chadwickbureau/baseballdatabank``
que usa pybaseball ya no existe, y la distribución oficial de SABR (Box) no
permite descarga programática. El paquete R ``Lahman`` en CRAN se mantiene
actualizado (v14 = temporada 2025) y CRAN está diseñado para descargas
automatizadas.

Requiere el grupo opcional ``data``: ``poetry install --with data``.
"""

import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from mlb_quant.ingestion.base import DEFAULT_TIMEOUT, DataSource, build_session
from mlb_quant.utils.frames import sanitize_columns

logger = logging.getLogger(__name__)

CRAN_INDEX_URL = "https://cran.r-project.org/web/packages/Lahman/index.html"
CRAN_TARBALL_URL = "https://cran.r-project.org/src/contrib/Lahman_{version}.tar.gz"

#: Versión usada si no se puede resolver la actual desde CRAN.
FALLBACK_VERSION = "14.0-0"


@dataclass(frozen=True)
class LahmanTable:
    """Descripción de una tabla Lahman soportada.

    Attributes:
        db_table: Nombre de la tabla destino en DuckDB.
        keys: Columnas (ya saneadas) que identifican una fila.
        rdata_name: Nombre del archivo ``.RData`` dentro del paquete.
    """

    db_table: str
    keys: tuple[str, ...]
    rdata_name: str


#: Tablas soportadas: nombre lógico -> destino, claves y archivo RData.
LAHMAN_TABLES: dict[str, LahmanTable] = {
    "people": LahmanTable("lahman_people", ("playerid",), "People"),
    "batting": LahmanTable("lahman_batting", ("playerid", "yearid", "stint"), "Batting"),
    "pitching": LahmanTable("lahman_pitching", ("playerid", "yearid", "stint"), "Pitching"),
    "fielding": LahmanTable(
        "lahman_fielding", ("playerid", "yearid", "stint", "pos"), "Fielding"
    ),
    "teams": LahmanTable("lahman_teams", ("yearid", "teamid"), "Teams"),
}


class LahmanSource(DataSource):
    """Acceso a las tablas de la Lahman Database (tarball de CRAN cacheado)."""

    source_name = "lahman"

    def __init__(
        self,
        cache_dir: Path,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT * 4,
    ) -> None:
        """Inicializa la fuente.

        Args:
            cache_dir: Directorio donde cachear el tarball descargado
                (p. ej. ``data/external/lahman``).
            session: Sesión HTTP inyectable (para tests).
            timeout: Timeout por petición (el tarball pesa ~7 MB).
        """
        self._cache_dir = cache_dir
        self._session = session or build_session()
        self._timeout = timeout

    def fetch_table(self, name: str) -> pd.DataFrame:
        """Devuelve una tabla Lahman con columnas saneadas.

        Descarga el tarball de CRAN en la primera llamada y lo cachea.

        Args:
            name: Nombre lógico (clave de :data:`LAHMAN_TABLES`).

        Returns:
            Tabla completa, columnas saneadas.

        Raises:
            ValueError: Si la tabla no está soportada o faltan claves.
        """
        spec = LAHMAN_TABLES.get(name)
        if spec is None:
            msg = f"Tabla Lahman no soportada: {name!r}. Opciones: {sorted(LAHMAN_TABLES)}"
            raise ValueError(msg)

        df = sanitize_columns(self._load_rdata(spec.rdata_name))
        missing = [k for k in spec.keys if k not in df.columns]
        if missing:
            msg = f"Tabla Lahman {name!r} sin columnas clave: {missing}"
            raise ValueError(msg)
        df = df.drop_duplicates(subset=list(spec.keys))
        logger.info("Lahman %s: %d filas.", name, len(df))
        return df

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #

    def _load_rdata(self, rdata_name: str) -> pd.DataFrame:
        """Extrae y lee un ``.RData`` del tarball (pyreadr: import perezoso)."""
        import pyreadr

        tarball = self._ensure_tarball()
        member = f"Lahman/data/{rdata_name}.RData"
        target = self._cache_dir / f"{rdata_name}.RData"
        if not target.exists():
            with tarfile.open(tarball, "r:gz") as tar:
                extracted = tar.extractfile(member)
                if extracted is None:
                    msg = f"Miembro {member!r} no encontrado en {tarball}."
                    raise ValueError(msg)
                target.write_bytes(extracted.read())
        result = pyreadr.read_r(str(target))
        df: pd.DataFrame = next(iter(result.values()))
        return df

    def _ensure_tarball(self) -> Path:
        """Devuelve la ruta del tarball, descargándolo si no está cacheado."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached = sorted(self._cache_dir.glob("Lahman_*.tar.gz"))
        if cached:
            return cached[-1]

        version = self._resolve_version()
        url = CRAN_TARBALL_URL.format(version=version)
        logger.info("Descargando Lahman %s desde CRAN...", version)
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        target = self._cache_dir / f"Lahman_{version}.tar.gz"
        target.write_bytes(response.content)
        return target

    def _resolve_version(self) -> str:
        """Lee la versión vigente del índice de CRAN; fallback a la fijada."""
        try:
            response = self._session.get(CRAN_INDEX_URL, timeout=self._timeout)
            response.raise_for_status()
            match = re.search(r"Lahman_([0-9][0-9.\-]*)\.tar\.gz", response.text)
            if match:
                return match.group(1)
        except requests.RequestException:
            logger.warning("No se pudo resolver la versión de CRAN; uso fallback.")
        return FALLBACK_VERSION
