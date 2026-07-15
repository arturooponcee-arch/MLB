"""Configuración de entorno del proyecto.

Separación de responsabilidades:
    - ``Settings`` (pydantic-settings): secretos y valores por máquina, vía ``.env``.
    - Hydra (``config/``): parámetros de pipelines y modelos, vía YAML.

Los secretos nunca van en YAML; los hiperparámetros nunca van en ``.env``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Valores de entorno y rutas base del proyecto.

    Todos los campos pueden sobrescribirse con variables de entorno o ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Rutas -------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    database_dir: Path = Field(default=PROJECT_ROOT / "database")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")
    outputs_dir: Path = Field(default=PROJECT_ROOT / "outputs")

    # --- Base de datos -----------------------------------------------------
    duckdb_path: Path = Field(default=PROJECT_ROOT / "database" / "mlb.duckdb")

    # --- Credenciales externas (fases posteriores) -------------------------
    odds_api_key: str = Field(default="", description="API key de The Odds API.")

    # --- Runtime -----------------------------------------------------------
    log_level: str = Field(default="INFO")

    @property
    def raw_dir(self) -> Path:
        """Directorio de datos crudos."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directorio de datos procesados."""
        return self.data_dir / "processed"

    @property
    def external_dir(self) -> Path:
        """Directorio de datos externos (Lahman, Retrosheet, etc.)."""
        return self.data_dir / "external"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia única de :class:`Settings` (cacheada)."""
    return Settings()
