"""Generador del dashboard HTML de juegos y probabilidades.

Página autocontenida (CSS inline, sin dependencias externas) que se
regenera con ``mlb report dashboard`` y se auto-recarga en el navegador
cada 60 s. Sin API de cuotas: en lugar de EV muestra la CUOTA JUSTA
(1/p) — una apuesta manual tiene valor solo si el book paga más.
"""

import html
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from mlb_quant.analysts.brain import DailyParlays

logger = logging.getLogger(__name__)

_REFRESH_SECONDS = 60


def render_dashboard(
    games: pl.DataFrame,
    target_date: str,
    output_path: Path,
    parlays: "DailyParlays | None" = None,
) -> Path:
    """Escribe el dashboard HTML.

    Args:
        games: Una fila por juego con metadatos (equipos, probables,
            ``game_datetime_utc``) y predicciones (``p_home_win``,
            ``sim_*``).
        target_date: Fecha mostrada en el encabezado (ISO).
        output_path: Ruta del HTML resultante.
        parlays: Combinadas del cerebro pronosticador (opcional).

    Returns:
        La ruta escrita.
    """
    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = _render_games(games) if not games.is_empty() else _render_empty()
    tiles = _render_tiles(games, target_date)
    parlays_html = _render_parlays(parlays) if parlays and parlays.parlays else ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _TEMPLATE.format(
            refresh=_REFRESH_SECONDS,
            generated=generated,
            target_date=target_date,
            tiles=tiles,
            body=body,
            parlays=parlays_html,
        ),
        encoding="utf-8",
    )
    logger.info("Dashboard escrito: %s (%d juegos).", output_path, len(games))
    return output_path


def _fair(p: float | None) -> str:
    """Cuota decimal justa: 1/p."""
    if p is None or p <= 0.001:
        return "—"
    return f"{1 / p:.2f}"


def _pct(p: float | None) -> str:
    return "—" if p is None else f"{100 * p:.1f}%"


def _bar(p: float | None, css_class: str) -> str:
    width = 0.0 if p is None else max(min(p, 1.0), 0.0) * 100
    return (
        f'<div class="bar-track"><div class="bar {css_class}" '
        f'style="width:{width:.1f}%"></div></div>'
    )


def _as_float(row: dict[str, object], key: str) -> float | None:
    """Extrae un valor numérico opcional de la fila."""
    value = row.get(key)
    return None if value is None else float(value)  # type: ignore[arg-type]


def _render_tiles(games: pl.DataFrame, target_date: str) -> str:
    n = len(games)
    avg_total = f"{float(games['sim_exp_total'].mean()):.1f}" if n else "—"  # type: ignore[arg-type]
    return f"""
    <div class="tiles">
      <div class="tile"><div class="tile-value">{html.escape(target_date)}</div>
        <div class="tile-label">Fecha de juegos</div></div>
      <div class="tile"><div class="tile-value">{n}</div>
        <div class="tile-label">Juegos programados</div></div>
      <div class="tile"><div class="tile-value">{avg_total}</div>
        <div class="tile-label">Total esperado promedio</div></div>
    </div>"""


def _render_empty() -> str:
    return '<p class="empty">Sin juegos programados en la ventana consultada.</p>'


def _render_parlays(parlays: "DailyParlays") -> str:
    """Tarjetas de combinadas sugeridas por el cerebro pronosticador."""
    cards = []
    for parlay in parlays.parlays:
        legs = "\n".join(
            f'<li>{html.escape(str(leg["selection"]))}'
            f' <span class="fair">(p {leg["p"]:.3f})</span></li>'
            for leg in parlay.legs.iter_rows(named=True)
        )
        name = html.escape(parlay.name.capitalize())
        odds = f"p {parlay.p_combined:.3f} · justa {parlay.fair_odds:.2f}"
        cards.append(f"""
      <div class="parlay">
        <div class="parlay-head"><span class="parlay-name">{name}</span>
          <span class="parlay-odds">{odds}</span></div>
        <ul class="parlay-legs">{legs}</ul>
      </div>""")
    return f"""
    <h2 class="section">Combinadas del día</h2>
    <div class="parlays">{"".join(cards)}</div>
    <p class="parlay-note">Una pierna por juego. La cuota justa es el umbral:
    la combinada solo tiene valor si el book paga más.</p>"""


def _render_games(games: pl.DataFrame) -> str:
    rows = "\n".join(_render_row(row) for row in games.iter_rows(named=True))
    return f"""
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Hora local</th><th>Juego</th>
        <th>P(visita)</th><th>P(local)</th>
        <th>Cuota justa<br>V / L</th>
        <th>Total esp.</th><th>Over 8.5</th>
        <th>RL local &minus;1.5</th><th>F5 local</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table></div>"""


def _render_row(row: dict[str, object]) -> str:
    away = html.escape(str(row.get("away_team_name") or "?"))
    home = html.escape(str(row.get("home_team_name") or "?"))
    away_sp = html.escape(str(row.get("away_probable_pitcher_name") or "por anunciar"))
    home_sp = html.escape(str(row.get("home_probable_pitcher_name") or "por anunciar"))
    start_utc = html.escape(str(row.get("game_datetime_utc") or ""))

    p_home = _as_float(row, "p_home_win")
    p_away = None if p_home is None else 1.0 - p_home
    p_over = _as_float(row, "sim_p_over")
    exp_total = _as_float(row, "sim_exp_total")
    p_runline = _as_float(row, "sim_p_home_runline")
    p_f5 = _as_float(row, "sim_p_f5_home_ml")

    return f"""<tr>
      <td class="time" data-utc="{start_utc}">—</td>
      <td class="matchup"><span class="team away-dot">{away}</span>
        <span class="sp">{away_sp}</span>
        <span class="at">@</span>
        <span class="team home-dot">{home}</span>
        <span class="sp">{home_sp}</span></td>
      <td class="num">{_pct(p_away)}{_bar(p_away, "away")}</td>
      <td class="num">{_pct(p_home)}{_bar(p_home, "home")}</td>
      <td class="num">{_fair(p_away)} / {_fair(p_home)}</td>
      <td class="num">{"—" if exp_total is None else f"{exp_total:.1f}"}</td>
      <td class="num">{_pct(p_over)} <span class="fair">({_fair(p_over)})</span></td>
      <td class="num">{_pct(p_runline)} <span class="fair">({_fair(p_runline)})</span></td>
      <td class="num">{_pct(p_f5)} <span class="fair">({_fair(p_f5)})</span></td>
    </tr>"""


_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Quant — Dashboard</title>
<style>
:root {{
  --surface: #fbfaf7; --surface-2: #ffffff; --border: #e4e2da;
  --text: #1a1a19; --text-2: #5f5e56; --muted: #8a887d;
  --home: #2a78d6; --away: #1baf7a; --track: #edece6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface: #1a1a19; --surface-2: #242422; --border: #3a3935;
    --text: #ffffff; --text-2: #c3c2b7; --muted: #8a887d;
    --home: #3987e5; --away: #199e70; --track: #33322f;
  }}
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--surface); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px; }}
header {{ display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
h1 {{ font-size: 18px; font-weight: 650; }}
.generated {{ color: var(--muted); font-size: 12px; }}
.tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.tile {{ background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 18px; min-width: 140px; }}
.tile-value {{ font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }}
.tile-label {{ color: var(--text-2); font-size: 12px; margin-top: 2px; }}
.table-wrap {{ overflow-x: auto; background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; min-width: 900px; }}
th {{ text-align: left; color: var(--text-2); font-size: 11px;
  text-transform: uppercase; letter-spacing: .04em; font-weight: 600;
  padding: 10px 12px; border-bottom: 1px solid var(--border); }}
td {{ padding: 10px 12px; border-bottom: 1px solid var(--border);
  vertical-align: middle; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: color-mix(in srgb, var(--track) 45%, transparent); }}
.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
.matchup {{ min-width: 230px; }}
.team {{ font-weight: 600; }}
.sp {{ display: block; color: var(--muted); font-size: 11px; margin: 0 0 4px 14px; }}
.at {{ color: var(--muted); margin-right: 4px; }}
.time {{ color: var(--text-2); white-space: nowrap; }}
.fair {{ color: var(--muted); font-size: 12px; }}
.bar-track {{ background: var(--track); border-radius: 4px; height: 6px;
  width: 90px; margin-top: 4px; }}
.bar {{ height: 6px; border-radius: 4px; }}
.bar.home {{ background: var(--home); }}
.bar.away {{ background: var(--away); }}
.home-dot::before, .away-dot::before {{ content: ""; display: inline-block;
  width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.home-dot::before {{ background: var(--home); }}
.away-dot::before {{ background: var(--away); }}
.empty {{ color: var(--text-2); padding: 32px; text-align: center; }}
.section {{ font-size: 15px; font-weight: 650; margin: 20px 0 10px; }}
.parlays {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.parlay {{ background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; min-width: 260px; flex: 1 1 260px; }}
.parlay-head {{ display: flex; justify-content: space-between; gap: 8px;
  margin-bottom: 8px; }}
.parlay-name {{ font-weight: 650; }}
.parlay-odds {{ color: var(--text-2); font-size: 12px;
  font-variant-numeric: tabular-nums; }}
.parlay-legs {{ margin: 0; padding-left: 18px; }}
.parlay-legs li {{ margin: 2px 0; }}
.parlay-note {{ color: var(--muted); font-size: 12px; margin-top: 8px;
  max-width: 72ch; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 14px; max-width: 72ch; }}
</style>
</head>
<body>
<header>
  <h1>MLB Quant — juegos y probabilidades</h1>
  <span class="generated">Generado {generated} · se recarga cada {refresh}s</span>
</header>
{tiles}
{body}
{parlays}
<footer>Probabilidad de moneyline: ensemble calibrado. Over/under, run line y F5:
Monte Carlo 10.000 sims sobre el modelo Poisson. La <b>cuota justa</b> (entre
paréntesis) es el umbral de valor: una apuesta solo tiene EV+ si el book paga MÁS
que esa cuota. Sin lineups confirmados las probabilidades usan imputación.</footer>
<script>
for (const cell of document.querySelectorAll(".time[data-utc]")) {{
  const iso = cell.dataset.utc;
  if (!iso) continue;
  const d = new Date(iso);
  if (!isNaN(d)) cell.textContent = d.toLocaleTimeString([],
    {{ hour: "2-digit", minute: "2-digit" }});
}}
</script>
</body>
</html>
"""
