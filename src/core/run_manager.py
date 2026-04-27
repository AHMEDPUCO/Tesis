from __future__ import annotations
import os, json, shutil
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict


RUNS_DIR_ENV = "CYBER_RANGE_RUNS_DIR"
_RESOLVED_RUNS_ROOT: str | None = None


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def _is_writable_directory(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def runs_root() -> str:
    global _RESOLVED_RUNS_ROOT
    if _RESOLVED_RUNS_ROOT:
        return _RESOLVED_RUNS_ROOT

    explicit = os.getenv(RUNS_DIR_ENV)
    candidates = [explicit] if explicit else []
    candidates.extend([
        os.path.join("data", "runs"),
        os.path.join("data", "experiments", "_runs"),
    ])

    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        path = os.path.normpath(raw)
        if path in seen:
            continue
        seen.add(path)
        if _is_writable_directory(path):
            _RESOLVED_RUNS_ROOT = path
            os.environ[RUNS_DIR_ENV] = path
            return path

    fallback = os.path.normpath(os.path.join("data", "experiments", "_runs"))
    os.makedirs(fallback, exist_ok=True)
    _RESOLVED_RUNS_ROOT = fallback
    os.environ[RUNS_DIR_ENV] = fallback
    return fallback


def run_dir(run_id: str) -> str:
    return os.path.join(runs_root(), run_id)


def run_paths(run_id: str) -> Dict[str, str]:
    base = run_dir(run_id)
    return {
        "base": base,
        "decisions": os.path.join(base, "decisions.jsonl"),
        "actions": os.path.join(base, "enforcement_actions.jsonl"),
        "meta": os.path.join(base, "run_meta.json"),
    }


def prepare_run(run_id: str, *, clean: bool = False, meta: dict | None = None) -> Dict[str, str]:
    paths = run_paths(run_id)
    base_exists = os.path.exists(paths["base"])

    if clean and base_exists:
        shutil.rmtree(paths["base"])
        base_exists = False  # ahora ya no existe

    os.makedirs(paths["base"], exist_ok=True)

    # ✅ Solo truncar si es run nuevo o si clean=True
    if clean or not base_exists:
        open(paths["decisions"], "w", encoding="utf-8").close()
        open(paths["actions"], "w", encoding="utf-8").close()
    else:
        # ✅ Asegura que existan sin borrar contenido
        open(paths["decisions"], "a", encoding="utf-8").close()
        open(paths["actions"], "a", encoding="utf-8").close()

    # ✅ Meta: solo crear si es nuevo o clean
    if clean or not os.path.exists(paths["meta"]):
        meta_out = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **(meta or {}),
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

    return paths

