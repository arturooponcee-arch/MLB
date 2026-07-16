"""Curvas de calibración (reliability diagrams) como SVG inline.

Mismo estilo que el dashboard: HTML autocontenido, CSS inline, sin
librerías de gráficos. Cada punto es un bin de ``calibration_table``
(p predicha vs. p observada, radio proporcional a sqrt(n)); la diagonal
es la calibración perfecta.
"""

import math

import polars as pl

_SIZE = 320
_MARGIN = 36
_MAX_RADIUS = 14.0
_MIN_RADIUS = 3.0


def calibration_svg(table: pl.DataFrame, title: str, size: int = _SIZE) -> str:
    """Dibuja una curva de calibración como SVG inline.

    Args:
        table: Salida de ``calibration_table`` (``p_predicted``,
            ``p_observed``, ``n`` por bin).
        title: Título del gráfico.
        size: Lado del SVG en px.

    Returns:
        Markup ``<svg>`` autocontenido.
    """
    plot = size - 2 * _MARGIN

    def x(p: float) -> float:
        return _MARGIN + p * plot

    def y(p: float) -> float:
        return size - _MARGIN - p * plot

    max_n = max(table["n"].max() or 1, 1)
    points = []
    for row in table.iter_rows(named=True):
        radius = _MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * math.sqrt(row["n"] / max_n)
        points.append(
            f'<circle cx="{x(row["p_predicted"]):.1f}" cy="{y(row["p_observed"]):.1f}" '
            f'r="{radius:.1f}" fill="#2f81f7" fill-opacity="0.65">'
            f"<title>pred {row['p_predicted']:.3f} / obs {row['p_observed']:.3f} "
            f"(n={row['n']})</title></circle>"
        )

    axis_labels = "".join(
        f'<text x="{x(tick):.0f}" y="{size - _MARGIN + 16}" text-anchor="middle" '
        f'class="tick">{tick:.1f}</text>'
        f'<text x="{_MARGIN - 8}" y="{y(tick) + 4:.0f}" text-anchor="end" '
        f'class="tick">{tick:.1f}</text>'
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">
  <style>.tick {{ font: 10px sans-serif; fill: #8b949e; }}
         .title {{ font: bold 13px sans-serif; fill: currentColor; }}
         .axis {{ stroke: #8b949e; stroke-width: 1; }}</style>
  <text x="{size / 2}" y="18" text-anchor="middle" class="title">{title}</text>
  <line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(0)}" class="axis"/>
  <line x1="{x(0)}" y1="{y(0)}" x2="{x(0)}" y2="{y(1)}" class="axis"/>
  <line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}"
        stroke="#8b949e" stroke-dasharray="4 3"/>
  {axis_labels}
  {"".join(points)}
</svg>"""


def render_calibration_html(
    per_season: dict[int, pl.DataFrame],
    column: str,
    metrics: dict[int, dict[str, float]],
) -> str:
    """Página HTML con una curva de calibración por temporada.

    Args:
        per_season: Tabla de calibración por temporada.
        column: Columna de probabilidad evaluada (para el encabezado).
        metrics: Métricas por temporada (``brier``, ``log_loss``, ``n``).

    Returns:
        Documento HTML autocontenido.
    """
    sections = []
    for season in sorted(per_season):
        stats = metrics.get(season, {})
        caption = (
            f"Brier {stats.get('brier', float('nan')):.4f} · "
            f"log-loss {stats.get('log_loss', float('nan')):.4f} · "
            f"n={int(stats.get('n', 0))}"
        )
        sections.append(
            '<figure class="card">'
            + calibration_svg(per_season[season], f"Temporada {season}")
            + f"<figcaption>{caption}</figcaption></figure>"
        )
    body = "".join(sections)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Calibración — {column}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328;
          background: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e6edf3; background: #0d1117; }}
    .card {{ border-color: #30363d; }}
  }}
  h1 {{ font-size: 1.3rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1.5rem; }}
  .card {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem;
           margin: 0; }}
  figcaption {{ font-size: 0.85rem; text-align: center; opacity: 0.8;
                margin-top: 0.5rem; }}
</style>
</head>
<body>
<h1>Curvas de calibración por temporada — <code>{column}</code></h1>
<p>Puntos sobre la diagonal = probabilidades honestas. Radio ∝ √n del bin.</p>
<div class="grid">{body}</div>
</body>
</html>"""
