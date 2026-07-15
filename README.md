# MLB Quant

Sistema profesional de análisis cuantitativo y pronóstico para MLB: estimación de
probabilidades multi-mercado, detección de apuestas EV+ y reportes automáticos.

## Filosofía

- **Interpretable primero**: modelos de conteo (Poisson/NegBin) como base; ML como mejora.
- **Modular**: un módulo por fuente de datos, por modelo y por mercado.
- **Reproducible**: configuración YAML (Hydra), secretos en `.env`, datos en DuckDB.

## Instalación

Requiere [Poetry](https://python-poetry.org) y Python 3.13.

```bash
poetry install            # núcleo + dev
poetry install --with ml,data,viz   # grupos opcionales por fase
cp .env.example .env
```

## Uso

```bash
poetry run mlb info       # verifica instalación y configuración
poetry run mlb daily run  # pipeline diario: ingesta -> features -> reportes
poetry run mlb bets scan  # cuotas en vivo vs. modelo -> apuestas EV+ (requiere ODDS_API_KEY)
```

## Estructura

```
config/          Configuración Hydra (parámetros, no secretos)
data/            raw / processed / external
database/        DuckDB warehouse
src/mlb_quant/   Código fuente (paquete instalable)
  ingestion/     Un módulo por fuente (MLB Stats API, Statcast, FanGraphs...)
  models/        Modelos individuales (Poisson, boosters...)
  simulations/   Monte Carlo
  betting/       EV, CLV, Kelly
tests/           pytest
```

## Desarrollo

```bash
poetry run pytest         # tests
poetry run ruff check .   # lint
poetry run mypy           # tipos
pre-commit install        # hooks de git
```
