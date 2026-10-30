"""Publish an exported GGUF to a private Hugging Face repo and deploy it with llm-model-manager.

llm-model-manager (github.com/kevin000001505/llm-model-manager) serves models it downloads from
Hugging Face, by repo and pinned revision, through llama-swap. So a fine-tune is uploaded to a
private repo on your account, then the manager's ``POST /api/setup`` is called with that repo;
the manager downloads it (with its own HF_TOKEN, which must be able to read your private repos)
and writes the llama-swap config. Callers on any machine then use llama-swap's OpenAI-compatible
API with the model id ``<config name>-<quant>``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import requests

from .progress import heartbeat

Log = Callable[[str], None]


def settings() -> dict:
    return {
        "token": os.environ.get("HF_TOKEN", "").strip(),
        "namespace": os.environ.get("SUBTUNE_HF_NAMESPACE", "").strip(),
        "manager_url": os.environ.get("LLM_MANAGER_URL", "").strip().rstrip("/"),
        "manager_gpu": os.environ.get("LLM_MANAGER_GPU", "").strip() or "all",
        "manager_params": os.environ.get("LLM_MANAGER_PARAMS", "").strip() or '{"c": 4096}',
        "swap_url": os.environ.get("LLAMA_SWAP_URL", "").strip().rstrip("/"),
    }


def missing_settings() -> list[str]:
    s = settings()
    missing = [] if s["token"] else ["HF_TOKEN (with write access)"]
    if not s["manager_url"]:
        missing.append("LLM_MANAGER_URL")
    return missing


def repo_name(tag: str) -> str:
    """Hugging Face repo name for a tag: letters, digits, '-', '_', '.'; no '--' or '..'."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{tag}-GGUF")
    return re.sub(r"([-.])\1+", r"\1", name)[:96]


def config_name(tag: str) -> str:
    """llm-model-manager config name (also the start of the llama-swap model id)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag.replace("__", "_"))


def upload(out_dir: Path, meta: dict, log: Log, api=None) -> dict:
    """Upload the run's GGUF and model card to a private repo; returns {"repo", "url", "file"}."""
    from huggingface_hub import HfApi

    s = settings()
    api = api or HfApi(token=s["token"])
    namespace = s["namespace"] or api.whoami()["name"]
    repo_id = f"{namespace}/{repo_name(meta['tag'])}"
    gguf = out_dir / meta["gguf"]["file"]
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
    log(f"Uploading {gguf.name} ({gguf.stat().st_size / 1e9:.1f} GB) to the private repo {repo_id}.")
    with heartbeat(log, "uploading to Hugging Face"):
        api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name, repo_id=repo_id,
                        commit_message=f"{meta['tag']} ({meta['gguf']['type']})")
    readme = out_dir / "README.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo_id)
    return {"repo": repo_id, "url": f"https://huggingface.co/{repo_id}", "file": gguf.name, "private": True}


def deploy(repo_id: str, meta: dict, log: Log) -> dict:
    """Ask llm-model-manager to download the repo and serve it through llama-swap."""
    s = settings()
    name, quant = config_name(meta["tag"]), meta["gguf"]["type"]
    body = {
        "hf_repo": repo_id,
        "quant": quant,
        "symlink_name": name,
        "parameters": s["manager_params"],
        "revision": "latest",
        "backend": "llama",
        "gpu": s["manager_gpu"],
    }
    resp = requests.post(f"{s['manager_url']}/api/setup", json=body, timeout=120)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text[:300]
        raise RuntimeError(f"llm-model-manager refused the deploy (HTTP {resp.status_code}): {detail}")
    status = resp.json().get("status", "")
    model_id = f"{name}-{quant}"
    log(f"llm-model-manager: {status} It serves the model as '{model_id}' once the download finishes.")
    return {"manager": s["manager_url"], "config": name, "model_id": model_id, "status": status}


def publish(out_dir: Path, meta: dict, log: Log) -> dict:
    """Upload, then deploy. Each step's failure is recorded rather than raised: the fine-tuned
    model itself is already saved locally."""
    result: dict = {}
    try:
        result.update(upload(out_dir, meta, log))
    except Exception as exc:
        log(f"Upload to Hugging Face failed: {exc}")
        return {"error": f"upload: {exc}"}
    if not settings()["manager_url"]:
        log("LLM_MANAGER_URL is not set: uploaded only. Deploy it from llm-model-manager's UI.")
        return result
    try:
        result["deployed"] = deploy(result["repo"], meta, log)
    except Exception as exc:
        log(f"Deploy to llm-model-manager failed: {exc}")
        result["deploy_error"] = str(exc)
    return result


def api_example(meta: dict) -> str:
    """curl call for the deployed model through llama-swap's OpenAI-compatible API."""
    deployed = (meta.get("published") or {}).get("deployed") or {}
    model_id = deployed.get("model_id") or f"{config_name(meta['tag'])}-{(meta.get('gguf') or {}).get('type', 'Q4_K_M')}"
    swap = settings()["swap_url"] or "http://<llama-swap host>:8080"
    body = {"model": model_id, "messages": [{"role": "user", "content": "Where are you going?"}], "temperature": 0}
    return f"curl {swap}/v1/chat/completions -H 'Content-Type: application/json' -d '{json.dumps(body, ensure_ascii=False)}'"
