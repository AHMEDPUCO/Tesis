from __future__ import annotations
import os, json, shutil
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def run_dir(run_id: str) -> str:
    return os.path.join("data", "runs", run_id)


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

