"""Criterio de Kelly y sizing de apuestas.

Sin API de cuotas: herramienta MANUAL. El usuario trae la cuota del book;
el sistema trae la probabilidad. Kelly fraccional (half/quarter) porque
el Kelly completo asume probabilidades exactas — las nuestras tienen
error de estimación, y sobre-apostar castiga más que sub-apostar.
"""

from dataclasses import dataclass


def expected_value(probability: float, decimal_odds: float) -> float:
    """EV por unidad apostada: ``p * cuota - 1``.

    Args:
        probability: Probabilidad estimada de acierto (0-1).
        decimal_odds: Cuota decimal del book (>1).

    Returns:
        EV por unidad (0.05 = +5%).
    """
    _validate(probability, decimal_odds)
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Fracción de Kelly completa: ``(p*b - q) / b`` con ``b = cuota - 1``.

    Args:
        probability: Probabilidad estimada de acierto (0-1).
        decimal_odds: Cuota decimal del book (>1).

    Returns:
        Fracción óptima del bankroll; 0 si no hay ventaja.
    """
    _validate(probability, decimal_odds)
    b = decimal_odds - 1.0
    fraction = (probability * b - (1.0 - probability)) / b
    return max(fraction, 0.0)


@dataclass(frozen=True)
class StakeRecommendation:
    """Recomendación de apuesta.

    Attributes:
        ev: EV por unidad apostada.
        fair_odds: Cuota justa (1/p): umbral mínimo de valor.
        kelly_full: Fracción de Kelly completa.
        stake: Monto recomendado (Kelly fraccional con tope).
        capped: Si el tope de bankroll recortó el stake.
    """

    ev: float
    fair_odds: float
    kelly_full: float
    stake: float
    capped: bool


def recommend_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    kelly_multiplier: float = 0.5,
    max_fraction: float = 0.05,
) -> StakeRecommendation:
    """Calcula el stake recomendado con Kelly fraccional y tope.

    Args:
        bankroll: Bankroll actual.
        probability: Probabilidad estimada de acierto (0-1).
        decimal_odds: Cuota decimal del book (>1).
        kelly_multiplier: Fracción de Kelly (0.5 = half, 0.25 = quarter).
        max_fraction: Tope duro como fracción del bankroll.

    Returns:
        Recomendación completa (stake 0 si no hay ventaja).

    Raises:
        ValueError: Con parámetros fuera de rango.
    """
    if bankroll <= 0:
        msg = "El bankroll debe ser positivo."
        raise ValueError(msg)
    if not 0 < kelly_multiplier <= 1:
        msg = "kelly_multiplier debe estar en (0, 1]."
        raise ValueError(msg)

    full = kelly_fraction(probability, decimal_odds)
    fraction = full * kelly_multiplier
    capped = fraction > max_fraction
    fraction = min(fraction, max_fraction)
    return StakeRecommendation(
        ev=expected_value(probability, decimal_odds),
        fair_odds=1.0 / probability,
        kelly_full=full,
        stake=round(bankroll * fraction, 2),
        capped=capped,
    )


def _validate(probability: float, decimal_odds: float) -> None:
    if not 0.0 < probability < 1.0:
        msg = f"Probabilidad fuera de (0, 1): {probability}"
        raise ValueError(msg)
    if decimal_odds <= 1.0:
        msg = f"La cuota decimal debe ser > 1: {decimal_odds}"
        raise ValueError(msg)
