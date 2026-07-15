"""Carga de configuración YAML (parámetros de pipelines).

Complementa a :mod:`mlb_quant.settings`: aquí viven hiperparámetros y
parámetros de pipeline (versionados en ``config/``); allí, secretos y rutas
por máquina (``.env``, no versionado).
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mlb_quant.settings import PROJECT_ROOT

CONFIG_DIR: Path = PROJECT_ROOT / "config"


def load_config(name: str = "config") -> DictConfig:
    """Carga un archivo YAML de ``config/`` como ``DictConfig``.

    Args:
        name: Nombre del archivo sin extensión.

    Returns:
        Configuración cargada.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        TypeError: Si el YAML raíz no es un mapeo.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        msg = f"El YAML raíz de {path} debe ser un mapeo, no {type(cfg).__name__}."
        raise TypeError(msg)
    return cfg
