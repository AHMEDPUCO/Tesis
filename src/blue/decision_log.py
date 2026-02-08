from __future__ import annotations
import json, os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def append_decision(
    *,
    episode_id: int,
    decision: str,                 # "block_ip" | "no_block" | "escalate"
    t_detect: Optional[str],        # ISO Z o None
    evidence: Dict[str, Any],
    reason: str,
    out_dir: str = "data/decisions",
    filename: str = "decisions.jsonl",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": _iso_now(),    # cuando se registró la decisión
        "episode_id": int(episode_id),
        "t_detect": t_detect,       # cuándo “detectó” el agente
        "decision": decision,
        "reason": reason,
        "evidence": evidence,
    }
    path = os.path.join(out_dir, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "path": path, "record": record}
