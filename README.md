# MLB Quant

Sistema profesional de análisis cuantitativo y pronóstico para MLB: estimación de
probabilidades multi-mercado (moneyline, totales, runline, F5, props), detección de
apuestas EV+ y reportes automáticos diarios.

## Filosofía

- **Interpretable primero**: modelos de conteo (Poisson) como base; ensemble calibrado como mejora.
- **Modular**: un módulo por fuente de datos, por modelo y por mercado.
- **Reproducible**: configuración YAML (Hydra), secretos en `.env`, datos en DuckDB.
- **Sin orquestador**: un comando idempotente + el programador del SO reemplazan a Prefect/Airflow.

## Estado actual

El pipeline completo funciona de punta a punta:

1. **Ingesta** — calendario, boxscores y lineups (MLB Stats API), pitch-by-pitch
   (Statcast), leaderboards (Baseball Savant), histórico (Lahman), clima por
   estadio (Open-Meteo) y, opcionalmente, cuotas (The Odds API).
2. **Features** — bloques de forma ofensiva, abridor, bullpen, parque, clima y
   lineup, tanto para juegos jugados (entrenamiento) como programados (predicción).
3. **Modelos** — Poisson de carreras (interpretable, con `mlb models explain`) +
   ensemble logístico calibrado para moneyline, validados con backtest walk-forward.
4. **Simulación** — Monte Carlo por juego (10k sims): totales, runline, F5.
5. **Props** — strikeouts del abridor (línea configurable).
6. **Apuestas** — cuota justa (1/p) en todos los reportes, EV y sizing con Kelly
   fraccional; escaneo automático del mercado si se configura una fuente de cuotas.
7. **Salidas** — dashboard HTML auto-actualizable y reportes diarios en
   Markdown/CSV/Excel (`reports/daily/`).

## Instalación

Requiere [Poetry](https://python-poetry.org) y Python 3.13.

```bash
poetry install            # núcleo + dev
poetry install --with ml,data,viz   # grupos opcionales por fase
cp .env.example .env      # opcional: solo si usarás una API de cuotas
```

## Uso

### Pipeline diario (lo habitual)

```bash
poetry run mlb daily run
```

Corre todo en orden: calendario → boxscores de ayer → Statcast → features →
clima → (cuotas si hay `ODDS_API_KEY`) → dashboard + reporte diario. Idempotente;
programable con el Task Scheduler de Windows (ver docstring de `mlb daily`).

### Comandos individuales

```bash
poetry run mlb info                    # verifica instalación y configuración
poetry run mlb ingest schedule --start 2026-07-01 --end 2026-07-15
poetry run mlb ingest boxscores --start 2026-07-14
poetry run mlb features build --start 2026-07-14
poetry run mlb models backtest        # walk-forward: Brier, log-loss, calibración
poetry run mlb models explain         # coeficientes del Poisson (efecto por σ)
poetry run mlb simulate run --date 2026-07-15
poetry run mlb props strikeouts --date 2026-07-15
poetry run mlb report daily           # juegos + props K en MD/CSV/Excel
poetry run mlb report dashboard       # dashboard HTML
```

### Apuestas: EV y Kelly

El flujo por defecto **no requiere ninguna API de cuotas**: los reportes imprimen
la cuota justa (1/p) de cada mercado; una apuesta solo tiene valor si tu book
paga más. Para dimensionarla:

```bash
poetry run mlb bets kelly --p 0.55 --odds 2.10 --bankroll 1000
```

Devuelve EV, cuota justa y stakes Kelly (full/half/quarter con tope duro).

Opcionalmente, con `ODDS_API_KEY` en `.env` (The Odds API), el escaneo es automático:

```bash
poetry run mlb ingest odds            # snapshot de cuotas -> odds_snapshots (base CLV)
poetry run mlb bets scan              # cuotas en vivo vs. modelo -> tabla EV+
```

## Estructura

```
config/            Configuración Hydra (parámetros, no secretos)
data/              raw / processed / external
database/          DuckDB warehouse (mlb.duckdb)
reports/daily/     Reportes diarios exportados
src/mlb_quant/
  ingestion/       Una fuente por módulo (MLB Stats API, Statcast, Savant,
                   Lahman, clima, The Odds API)
  preprocessing/   Agregados intermedios (batter_games)
  feature_engineering/  Bloques de features + features de juegos futuros
  models/          Poisson, ensemble calibrado, props
  simulations/     Monte Carlo multi-mercado
  evaluation/      Backtest walk-forward y métricas
  betting/         Detección de valor: mejor precio, EV, emparejado con cuotas
  bankroll/        Criterio de Kelly y sizing
  reporting/       Reporte diario + exports (MD/CSV/Excel)
  visualization/   Dashboard HTML
  cli/             CLI `mlb` (typer): ingest, features, models, simulate,
                   props, bets, report, daily
tests/             pytest (unitarios sin red; integración con -m integration)
```

## Desarrollo

```bash
poetry run pytest -m "not integration"  # tests sin red
poetry run ruff check .   # lint
poetry run mypy           # tipos
pre-commit install        # hooks de git
```
