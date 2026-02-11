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
    ###
    run_id: Optional[str] = None,
    out_path: Optional[str] = None,
    timestamp: Optional[str] = None,
    ######
    #out_dir: str = "data/decisions",
    #filename: str = "decisions.jsonl",
) -> Dict[str, Any]:
    ts = timestamp or _iso_now()
    out_path = out_path or os.path.join("data", "decisions", "decisions.jsonl")
    #os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": ts,    # cuando se registró la decisión
        "episode_id": int(episode_id),
        "run_id": run_id,
        "t_detect": t_detect,       # cuándo “detectó” el agente
        "decision": decision,
        "reason": reason,
        "evidence": evidence,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #path = out_path
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "path": out_path, "record": record}
