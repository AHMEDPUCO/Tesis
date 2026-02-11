from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def block_ip(
    ip: str,
    duration_seconds: int = 3600,
    *,
    episode_id: Optional[int] = None,
    run_id: Optional[str] = None,
    out_path: Optional[str] = None,
    reason: Optional[str] = None,
    out_dir: str = "data/actions",
    action_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: block_ip(ip, duration)
    SIMULADA: no aplica firewall real.
    Solo registra la acción para auditoría y para que el Judge/experimentos la usen.
    """
    # ✅ Respeta out_dir si no te pasan out_path
    out_path = out_path or os.path.join(out_dir, "enforcement_actions.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    action = {
        "timestamp": action_time or _iso_now(),
        "run_id": run_id,                          # ✅ IMPORTANTE
        "episode_id": int(episode_id) if episode_id is not None else None,
        "action": "block_ip",
        "ip": ip,
        "duration_seconds": int(duration_seconds),
        "reason": reason,
        "status": "simulated",
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(action, ensure_ascii=False) + "\n")

    return {"ok": True, "recorded_to": out_path, "action": action}

###python -m src.tools.enforcement