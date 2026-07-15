"""Tests de exportación multi-formato."""

from pathlib import Path

import polars as pl

from mlb_quant.reporting.exports import export_frame, to_markdown


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {"equipo": ["Yankees", "Red|Sox"], "p_home": [0.581, 0.412]}
    )


def test_export_all_formats(tmp_path: Path) -> None:
    written = export_frame(_df(), tmp_path / "reporte")
    assert {p.suffix for p in written} == {".csv", ".xlsx", ".md"}
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_csv_roundtrip(tmp_path: Path) -> None:
    written = export_frame(_df(), tmp_path / "r", formats=("csv",))
    back = pl.read_csv(written[0])
    assert back["p_home"].to_list() == [0.581, 0.412]


def test_markdown_table() -> None:
    md = to_markdown(_df())
    assert "| equipo | p_home |" in md
    assert "0.581" in md
    assert "Red\\|Sox" in md  # pipe escapado


def test_markdown_empty() -> None:
    assert "Sin datos" in to_markdown(pl.DataFrame())


def test_markdown_caps_rows() -> None:
    big = pl.DataFrame({"x": list(range(1000))})
    md = to_markdown(big, max_rows=10)
    assert md.count("\n") <= 13  # header + divider + 10 filas + margen
