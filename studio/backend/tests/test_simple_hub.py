# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from core.hub.simple_hub import download_local_model, list_local_gguf, load_local_model, select_local_model


def test_list_local_gguf_finds_files_and_ignores_companions(tmp_path: Path):
    (tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    (tmp_path / "mmproj.gguf").write_bytes(b"GGUF")
    (tmp_path / "notes.txt").write_text("nope")
    models = list_local_gguf(tmp_path)
    names = [item["name"] for item in models]
    assert "Qwen3.8-27B-UD-Q4_K_XL.gguf" in names
    assert "mmproj.gguf" not in names


def test_select_local_model_requires_an_existing_gguf(tmp_path: Path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    selected = select_local_model(str(path))
    assert selected["ok"] is True
    assert selected["path"].endswith("model.gguf")
    missing = select_local_model(str(tmp_path / "absent.gguf"))
    assert missing["ok"] is False


def test_download_stages_a_local_gguf(tmp_path: Path):
    src = tmp_path / "src" / "model.gguf"
    src.parent.mkdir()
    src.write_bytes(b"GGUF")
    dest = tmp_path / "dest"
    result = download_local_model(str(src), dest)
    assert result["ok"] is True
    assert result["downloaded"] is True
    assert (dest / "model.gguf").is_file()


def test_load_local_model_hands_resolved_gguf_to_inference_load(tmp_path: Path):
    path = tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    path.write_bytes(b"GGUF")
    seen: list[str] = []

    def load_fn(resolved: str):
        seen.append(resolved)
        return {"ok": True, "model": "resident-qwen", "status": "loaded"}

    result = load_local_model(str(path), load_fn=load_fn)
    assert result["ok"] is True
    assert result["loaded"] is True
    assert result["model"] == "resident-qwen"
    assert seen == [str(path.resolve())]


def test_hub_load_route_drives_inference_load_model_gated(tmp_path: Path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from routes import helix_engine

    gguf = tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    gguf.write_bytes(b"GGUF")
    seen: dict[str, object] = {}

    async def fake_gated(request, fastapi_request, current_subject, user_initiated=False):
        seen["model_path"] = request.model_path
        seen["user_initiated"] = user_initiated
        seen["subject"] = current_subject
        return SimpleNamespace(model=request.model_path, status="loaded")

    monkeypatch.setattr("routes.inference.load_model_gated", fake_gated)
    result = asyncio.run(
        helix_engine.hub_load(
            {"path": str(gguf)},
            fastapi_request=SimpleNamespace(),
            current_subject="tester",
        )
    )
    assert result["loaded"] is True
    assert seen["model_path"] == str(gguf.resolve())
    assert seen["user_initiated"] is True
    assert seen["subject"] == "tester"
