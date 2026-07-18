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


def _reload_api(tmp_path, monkeypatch, key: str = "test-secret-key-xyz"):
    import importlib

    import vtclone.api as api_mod

    key_file = tmp_path / "key"
    key_file.write_text(key + "\n", encoding="utf-8")
    monkeypatch.setenv("VTCLONE_API_KEY", key)
    monkeypatch.setenv("VTCLONE_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("VTCLONE_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VTCLONE_JOBS_DIR", str(tmp_path / "jobs"))
    return importlib.reload(api_mod)


def test_public_api_key_auth(tmp_path, monkeypatch):
    """Unauthenticated /v1 fails; X-API-Key and Bearer work; /health is open."""
    api_mod = _reload_api(tmp_path, monkeypatch)
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


def test_multipart_video_upload_with_request(tmp_path, monkeypatch):
    """POST /v1/jobs must accept the video file in the same multipart request."""
    api_mod = _reload_api(tmp_path, monkeypatch)

    def fake_run(cfg, progress=None):
        from vtclone.pipeline import PipelineResult

        out = Path(cfg.project_dir) / "translated.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-video")
        segs = Path(cfg.project_dir) / "segments.json"
        segs.write_text('{"segments":[]}', encoding="utf-8")
        return PipelineResult(
            success=True,
            output_video=str(out),
            output_json=str(segs),
            project_dir=str(cfg.project_dir),
            stages_run=["all"],
            elapsed_sec=0.01,
        )

    api_mod.run_pipeline = fake_run
    from fastapi.testclient import TestClient

    client = TestClient(api_mod.create_app())
    headers = {"X-API-Key": "test-secret-key-xyz"}
    payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

    # Missing file → 400
    r = client.post(
        "/v1/jobs",
        headers=headers,
        data={"src_lang": "de", "mt_model": "Helsinki-NLP/opus-mt-de-en"},
    )
    assert r.status_code == 400
    assert "video" in r.json()["detail"].lower()

    # Field name video
    r = client.post(
        "/v1/jobs",
        headers=headers,
        files={"video": ("clip.mp4", payload, "video/mp4")},
        data={"src_lang": "de", "mt_model": "Helsinki-NLP/opus-mt-de-en"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] in ("queued", "running", "completed")
    assert Path(body["input_video"]).is_file()
    assert Path(body["input_video"]).read_bytes() == payload

    # Alias field name file
    r = client.post(
        "/v1/jobs",
        headers=headers,
        files={"file": ("other.mov", payload, "video/quicktime")},
        data={"src_lang": "ru"},
    )
    assert r.status_code == 202, r.text


def test_binary_body_video_upload(tmp_path, monkeypatch):
    api_mod = _reload_api(tmp_path, monkeypatch)
    api_mod.run_pipeline = lambda cfg, progress=None: __import__(
        "vtclone.pipeline", fromlist=["PipelineResult"]
    ).PipelineResult(success=True, stages_run=[], elapsed_sec=0.0)

    from fastapi.testclient import TestClient

    client = TestClient(api_mod.create_app())
    payload = b"\x00\x00\x00\x18ftypmp42" + b"\x11" * 32
    r = client.post(
        "/v1/jobs/binary",
        params={"src_lang": "de", "mt_model": "Helsinki-NLP/opus-mt-de-en"},
        headers={
            "X-API-Key": "test-secret-key-xyz",
            "Content-Type": "video/mp4",
        },
        content=payload,
    )
    assert r.status_code == 202, r.text
    assert Path(r.json()["input_video"]).read_bytes() == payload
