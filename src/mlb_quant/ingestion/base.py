"""Infraestructura común de ingesta.

Contrato: cada fuente es una clase que expone métodos ``fetch_*`` que
devuelven DataFrames ya tipados. La persistencia NO vive aquí — es
responsabilidad de :class:`mlb_quant.db.Database` (SRP).
"""

from abc import ABC
from typing import ClassVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT: float = 30.0
_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


class DataSource(ABC):
    """Base de toda fuente de datos.

    Attributes:
        source_name: Identificador único de la fuente (para logs y metadatos).
    """

    source_name: ClassVar[str]


def build_session(max_retries: int = 3, backoff_seconds: float = 2.0) -> requests.Session:
    """Crea una sesión HTTP con reintentos exponenciales.

    Reintenta ante errores transitorios (429, 5xx) respetando los rate
    limits de las fuentes.

    Args:
        max_retries: Número máximo de reintentos por petición.
        backoff_seconds: Factor de espera exponencial entre reintentos.

    Returns:
        Sesión configurada.
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_seconds,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
