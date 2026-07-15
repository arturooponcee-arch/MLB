"""Tests de mlb_quant.settings."""

from pathlib import Path

from mlb_quant.settings import PROJECT_ROOT, Settings


def test_project_root_is_repo_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_default_paths_derive_from_root() -> None:
    settings = Settings(_env_file=None)
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.processed_dir == PROJECT_ROOT / "data" / "processed"
    assert settings.external_dir == PROJECT_ROOT / "data" / "external"
    assert settings.duckdb_path.suffix == ".duckdb"


def test_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATA_DIR", str(Path("/tmp/otro")))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.data_dir == Path("/tmp/otro")
    assert settings.log_level == "DEBUG"
