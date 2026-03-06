from __future__ import annotations
from datetime import timedelta
from src.backend_a.search_logs import _parse_iso_z
import os, re
from typing import Dict, Any, Optional, List
from urllib import parse


from src.backend_a.search_logs import search_logs
from src.tools.asset_context import get_asset_context
from src.tools.enforcement import block_ip
from src.blue.decision_log import append_decision
from src.judge.judge_mtd_mttr import judge_mttd_mttr

DEFAULT_LOGS_DIR = "data/logs_backend_a"

def _list_episode_ids(logs_dir: str) -> List[int]:
    ids = []
    for fn in os.listdir(logs_dir):
        m = re.match(r"episode_(\d+)\.jsonl$", fn)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def _pick_detection_event(logs_dir: str, episode_id: int) -> Optional[Dict[str, Any]]:
    # Heurística: señales “sospechosas” (ajusta tags según tu generador)
    candidates = []

    for tag in ["suspicious", "lateral_like", "success_after_fail", "post_auth", "burst"]:
        r = search_logs(
            logs_dir,
            episode_id=episode_id,
            filters={"tags_any": [tag]},
            limit=50
        )
        candidates.extend(r["events"])

    if not candidates:
        return None

    # el más temprano por timestamp
    candidates.sort(key=lambda e: e["timestamp"])
    return candidates[0]

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=str, default=DEFAULT_LOGS_DIR)
    ap.add_argument("--gt-dir", type=str, default="data/ground_truth")
    ap.add_argument("--decisions-path", type=str, default="data/decisions/decisions.jsonl")
    ap.add_argument("--actions-path", type=str, default="data/actions/enforcement_actions.jsonl")
    ap.add_argument("--out-csv", type=str, default="data/results/results_phase2.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.decisions_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.actions_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    logs_dir = args.logs_dir
    episode_ids = _list_episode_ids(logs_dir)
    print("Episodes:", episode_ids)

    for ep in episode_ids:
        ev = _pick_detection_event(logs_dir, ep)

        if ev is None:
            append_decision(
                episode_id=ep,
                decision="no_block",
                t_detect=None,
                evidence={"matched": 0},
                out_path=args.decisions_path,
                reason="No se encontraron señales con tags sospechosos en este episodio."
            )
            continue

        # Enrich con asset context
        asset_ctx = get_asset_context(ev["host"])

        src_ip = ev.get("src_ip")
        t_detect = ev["timestamp"]

        # Regla simple: si severity medium/high y NO es allowlisted_user -> bloquear
        tags = ev.get("tags") or []
        sev = ev.get("severity", "low")
        allowlisted = "allowlisted_user" in tags or "service_account" in tags

        if src_ip and (sev in ("medium","high")) and not allowlisted:
            # simular 30s de demora
            t_respond = (_parse_iso_z(t_detect) + timedelta(seconds=30)).isoformat().replace("+00:00","Z")
            action = block_ip(
                src_ip,
                1800,
                episode_id=ep,
                reason="baseline: suspicious tags + severity",
                action_time=t_respond,
                out_path=args.actions_path,
            )
            append_decision(
                episode_id=ep,
                decision="block_ip",
                t_detect=t_detect,
                evidence={"event": ev, "asset_context": asset_ctx, "enforcement": action},
                reason="Baseline bloquea si (medium/high) y no allowlisted.",
                out_path=args.decisions_path,
            )
        else:
            append_decision(
                episode_id=ep,
                decision="no_block",
                t_detect=t_detect,
                evidence={"event": ev, "asset_context": asset_ctx},
                reason="Baseline NO bloquea (allowlisted o severidad low).",
                out_path=args.decisions_path,
            )

    out_csv = judge_mttd_mttr(
        gt_dir=args.gt_dir,
        decisions_path=args.decisions_path,
        actions_path=args.actions_path,
        out_csv=args.out_csv,
    )
    print("Results:", out_csv)

if __name__ == "__main__":
    main()
