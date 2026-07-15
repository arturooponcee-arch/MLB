"""Configuración central de logging.

Consola con Rich (colores, trazas legibles) + archivo rotativo en ``logs/``.
Cualquier módulo obtiene su logger con ``logging.getLogger(__name__)``.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logging(level: str = "INFO", logs_dir: Path | None = None) -> None:
    """Configura el logger raíz del proyecto.

    Idempotente: llamadas repetidas no duplican handlers.

    Args:
        level: Nivel mínimo de log (``DEBUG``, ``INFO``, ...).
        logs_dir: Directorio para el archivo de log. Si es ``None``,
            solo se registra en consola.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    console = RichHandler(rich_tracebacks=True, show_path=False)
    console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(console)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / "mlb_quant.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
