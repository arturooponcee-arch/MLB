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
2. **Features** — bloques de forma ofensiva, abridor (resultados + Statcast:
   whiff%, CSW%, velocidad de recta, xwOBA en contra), bullpen, parque, clima,
   lineup, platoon (equipo vs. mano del abridor rival), Elo y descanso/viaje;
   tanto para juegos jugados (entrenamiento) como programados (predicción).
3. **Modelos** — Poisson de carreras (interpretable, con `mlb models explain`) +
   ensemble calibrado (Platt por defecto; isotónica/beta/stacking comparables
   con `mlb models backtest --compare`), validados con backtest walk-forward y
   curvas de calibración por temporada (`mlb models calibration`).
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

### Despliegue en Vercel

El pipeline diario despliega el dashboard estático (`index.html` + `daily/`)
al proyecto Vercel `mlb-quant-dashboard` (team `arturoo10`) en cada corrida,
vía `vercel deploy --prod` con la CLI — sin ramas intermedias.

Requiere un secret `VERCEL_TOKEN` en el repo (Settings → Secrets and
variables → Actions). Genera el token en vercel.com/account/tokens y súbelo
con:

```bash
gh secret set VERCEL_TOKEN
```

- El mismo workflow también sigue publicando el dashboard vía GitHub Pages.

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

### Apuestas: EV y Kelly (sin API de cuotas)

El flujo por defecto **no requiere ninguna API**. Camino recomendado — cuotas
manuales por CSV:

```bash
poetry run mlb bets template                     # CSV con los juegos del día
# ... rellenas odds_away / odds_home con las cuotas de tu book ...
poetry run mlb bets scan --odds-file reports/daily/<fecha>_cuotas.csv
```

`scan` cruza tus cuotas con el modelo, tabula las apuestas EV+ con stake Kelly
y **persiste el snapshot en `odds_snapshots`**: aunque la fuente sea manual, el
warehouse acumula el histórico del mercado — la base para medir CLV y para
contrastar el modelo contra el mercado con el tiempo. Varias casas: repite la
fila del juego cambiando la columna `book` (el escaneo toma el mejor precio).

Para una apuesta suelta, sin CSV:

```bash
poetry run mlb bets kelly --p 0.55 --odds 2.10 --bankroll 1000
```

Devuelve EV, cuota justa y stakes Kelly (full/half/quarter con tope duro).

### Ledger y CLV (medir si el edge es real)

```bash
poetry run mlb bets log --date 2026-07-17 --team "Dodgers" --odds 2.05 --stake 25
poetry run mlb bets settle    # resultados + CLV vs. último snapshot pre-juego
poetry run mlb bets report    # ROI, aciertos y CLV medio del ledger
```

El CLV (cuota tomada vs. cierre del mercado) distingue habilidad de suerte en
decenas de apuestas; el ROI solo lo hace en miles. Requiere ir acumulando
snapshots de cuotas (CSV manual o API) antes de cada juego.

Opcionalmente, con `ODDS_API_KEY` en `.env` (The Odds API), el escaneo baja las
cuotas solo:

```bash
poetry run mlb ingest odds            # snapshot de cuotas -> odds_snapshots
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
