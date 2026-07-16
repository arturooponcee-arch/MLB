"""Calibradores de probabilidad intercambiables.

Un calibrador aprende el mapa ``probabilidad cruda -> probabilidad
calibrada`` sobre datos que el modelo base no vio. Variantes:

- ``isotonic``: regresión isotónica (no paramétrica, escalonada). Flexible
  pero cuantiza las probabilidades en mesetas.
- ``platt``: regresión logística sobre el logit (2 parámetros). Suave,
  robusta con pocos datos, asume distorsión sigmoide.
- ``beta``: calibración beta de Kull et al. 2017 — logística sobre
  ``[ln p, -ln(1-p)]`` (3 parámetros). Suave y más flexible que Platt.
  Nota: no impone la restricción de monotonía a, b >= 0 del paper
  (sklearn no lo permite); con colas de calibración de cientos de puntos
  y scores crudos aproximadamente monótonos no es un problema práctico.
- ``none``: identidad con clip (para cabezas ya calibradas, p. ej.
  stacking logístico).
"""

from typing import Literal, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

type CalibrationKind = Literal["isotonic", "platt", "beta", "none"]

#: Rango de salida de todo calibrador: evita probabilidades 0/1 exactas.
_OUTPUT_MIN = 0.01
_OUTPUT_MAX = 0.99
#: Clip interno de la probabilidad cruda antes de transformar.
_INPUT_EPS = 1e-6


class Calibrator(Protocol):
    """Contrato mínimo de un calibrador de probabilidades."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        """Ajusta el mapa con probabilidades crudas y resultados 0/1."""
        ...

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Aplica el mapa; salida en [0.01, 0.99]."""
        ...


class IsotonicCalibrator:
    """Regresión isotónica (comportamiento histórico del ensemble)."""

    def __init__(self) -> None:
        """Crea el calibrador sin ajustar."""
        self._model = IsotonicRegression(
            y_min=_OUTPUT_MIN, y_max=_OUTPUT_MAX, out_of_bounds="clip"
        )

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Ajusta la isotónica sobre (probabilidad cruda, resultado)."""
        self._model.fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Aplica el mapa isotónico (ya acotado a [0.01, 0.99])."""
        return np.asarray(self._model.predict(p), dtype=np.float64)


class PlattCalibrator:
    """Escalado de Platt: logística univariada sobre el logit crudo."""

    def __init__(self) -> None:
        """Crea el calibrador sin ajustar."""
        self._model = LogisticRegression(C=1e6, max_iter=2000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Ajusta la logística sobre el logit de la probabilidad cruda."""
        self._model.fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Aplica el sigmoide ajustado, con clip a [0.01, 0.99]."""
        raw = self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.asarray(np.clip(raw, _OUTPUT_MIN, _OUTPUT_MAX), dtype=np.float64)


class BetaCalibrator:
    """Calibración beta (Kull 2017): logística sobre ``[ln p, -ln(1-p)]``."""

    def __init__(self) -> None:
        """Crea el calibrador sin ajustar."""
        self._model = LogisticRegression(C=1e6, max_iter=2000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        """Ajusta la logística sobre las features beta de la probabilidad."""
        self._model.fit(_beta_features(p), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Aplica el mapa beta ajustado, con clip a [0.01, 0.99]."""
        raw = self._model.predict_proba(_beta_features(p))[:, 1]
        return np.asarray(np.clip(raw, _OUTPUT_MIN, _OUTPUT_MAX), dtype=np.float64)


class IdentityCalibrator:
    """Identidad con clip: para cabezas que ya emiten probabilidad calibrada."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        """No aprende nada; existe por compatibilidad de contrato."""
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad tal cual, con clip a [0.01, 0.99]."""
        return np.asarray(
            np.clip(np.asarray(p, dtype=np.float64), _OUTPUT_MIN, _OUTPUT_MAX),
            dtype=np.float64,
        )


def make_calibrator(kind: CalibrationKind) -> Calibrator:
    """Fábrica de calibradores por nombre.

    Args:
        kind: ``isotonic`` | ``platt`` | ``beta`` | ``none``.

    Returns:
        Calibrador sin ajustar.

    Raises:
        ValueError: Con un tipo desconocido.
    """
    factories: dict[str, type] = {
        "isotonic": IsotonicCalibrator,
        "platt": PlattCalibrator,
        "beta": BetaCalibrator,
        "none": IdentityCalibrator,
    }
    if kind not in factories:
        msg = f"Calibrador desconocido: {kind!r}. Opciones: {sorted(factories)}"
        raise ValueError(msg)
    calibrator: Calibrator = factories[kind]()
    return calibrator


def _clip_probability(p: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.clip(np.asarray(p, dtype=np.float64), _INPUT_EPS, 1.0 - _INPUT_EPS),
        dtype=np.float64,
    )


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = _clip_probability(p)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def _beta_features(p: np.ndarray) -> np.ndarray:
    clipped = _clip_probability(p)
    return np.column_stack([np.log(clipped), -np.log(1.0 - clipped)])
