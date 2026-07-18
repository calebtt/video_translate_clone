"""Unit tests that do not require GPU or model weights."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vtclone.utils import (
    atempo_filter_chain,
    fix_overlapping_segments,
    load_segments,
    save_segments,
)


def test_atempo_chain_within_range():
    assert atempo_filter_chain(1.0) == "atempo=1.000000"
    assert "atempo=" in atempo_filter_chain(1.2)
    chain = atempo_filter_chain(3.0)
    assert chain.count("atempo=") >= 2  # 2.0 * 1.5


def test_atempo_chain_slow():
    chain = atempo_filter_chain(0.3)
    assert "atempo=" in chain
    # Should split into 0.5 * 0.6
    assert chain.count("atempo=") >= 2


def test_fix_overlaps_delay():
    segs = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "a"},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "b"},
    ]
    fixed = fix_overlapping_segments(segs, min_gap=0.1, max_delay=5.0)
    assert fixed[1]["start"] >= fixed[0]["end"] + 0.1 - 1e-6
    # duration of second segment preserved when delaying
    assert abs((fixed[1]["end"] - fixed[1]["start"]) - 1.5) < 1e-6


def test_fix_overlaps_no_change():
    segs = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "a"},
        {"id": 1, "start": 1.5, "end": 2.5, "text": "b"},
    ]
    fixed = fix_overlapping_segments(segs, min_gap=0.1)
    assert fixed[0]["end"] == 1.0
    assert fixed[1]["start"] == 1.5


def test_save_load_segments(tmp_path: Path):
    segs = [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "translated_text": "hi"}]
    p = tmp_path / "segments.json"
    save_segments(p, segs)
    loaded = load_segments(p)
    assert loaded[0]["text"] == "hi"
    raw = json.loads(p.read_text())
    assert "segments" in raw


def test_cli_help():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        ["python", str(root / "video_translate_clone_perf.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root)},
    )
    assert r.returncode == 0
    assert "--match-duration" in r.stdout or "--match_duration" in r.stdout or "match" in r.stdout


def test_api_import():
    from vtclone.api import create_app

    a = create_app()
    assert a.title
    routes = {getattr(r, "path", None) for r in a.routes}
    assert "/health" in routes
    assert "/v1/jobs" in routes


def test_public_api_key_auth(tmp_path, monkeypatch):
    """Unauthenticated /v1 fails; X-API-Key and Bearer work; /health is open."""
    import importlib

    import vtclone.api as api_mod

    key_file = tmp_path / "key"
    key_file.write_text("test-secret-key-xyz\n", encoding="utf-8")
    monkeypatch.setenv("VTCLONE_API_KEY", "test-secret-key-xyz")
    monkeypatch.setenv("VTCLONE_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("VTCLONE_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VTCLONE_JOBS_DIR", str(tmp_path / "jobs"))

    importlib.reload(api_mod)
    from fastapi.testclient import TestClient

    client = TestClient(api_mod.create_app())

    assert client.get("/health").status_code == 200
    assert client.get("/v1/jobs").status_code == 401
    assert client.get("/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.get("/v1/jobs", headers={"X-API-Key": "test-secret-key-xyz"}).status_code
        == 200
    )
    assert (
        client.get(
            "/v1/jobs", headers={"Authorization": "Bearer test-secret-key-xyz"}
        ).status_code
        == 200
    )
