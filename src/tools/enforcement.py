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
    reason: Optional[str] = None,
    out_dir: str = "data/actions",
    action_time: str | None = None,
) -> Dict[str, Any]:
    """
    Tool: block_ip(ip, duration)
    SIMULADA: no aplica firewall real.
    Solo registra la acción para auditoría y para que el Judge/experimentos la usen.
    """
    os.makedirs(out_dir, exist_ok=True)

    action = {
        "timestamp": action_time or _iso_now(),
        "action": "block_ip",
        "ip": ip,
        "duration_seconds": int(duration_seconds),
        "episode_id": episode_id,
        "reason": reason,
        "status": "simulated"
    }

    path = os.path.join(out_dir, "enforcement_actions.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(action, ensure_ascii=False) + "\n")

    return {"ok": True, "recorded_to": path, "action": action}


if __name__ == "__main__":
    res = block_ip("192.168.1.50", 1800, episode_id=2, reason="suspicious auth_fail burst")
    print(json.dumps(res, indent=2, ensure_ascii=False))
###python -m src.tools.enforcement