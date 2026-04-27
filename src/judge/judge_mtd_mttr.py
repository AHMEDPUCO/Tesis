from __future__ import annotations
import csv, glob, json, os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

def _parse_iso_z(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _delta_seconds(start_ts: Optional[str], end_ts: Optional[str]) -> Optional[float]:
    if not start_ts or not end_ts:
        return None
    try:
        return (_parse_iso_z(end_ts) - _parse_iso_z(start_ts)).total_seconds()
    except Exception:
        return None

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _load_ground_truth(gt_dir: str, episode_id: int) -> Optional[Dict[str, Any]]:
    # soporta varios nombres
    candidates = [
        os.path.join(gt_dir, f"episode_{episode_id:03d}.json"),
        os.path.join(gt_dir, f"episode_{episode_id}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def _get_injection_start(gt: Dict[str, Any]) -> Optional[str]:
    # Si el episodio es benigno, no tiene sentido medir MTTD/MTTR contra inyeccion.
    if gt.get("attack_present") is False:
        return None
    # intenta varias llaves comunes
    if "injected_window" in gt and isinstance(gt["injected_window"], dict):
        return gt["injected_window"].get("start")
    if "window" in gt and isinstance(gt["window"], dict):
        return gt["window"].get("start")
    return gt.get("t0")

def _first_block_time(actions: List[Dict[str, Any]], episode_id: int) -> Optional[str]:
    blocks = [
        a for a in actions
        if a.get("action") == "block_ip" and a.get("episode_id") == episode_id
    ]
    if not blocks:
        return None
    blocks.sort(key=lambda x: x.get("timestamp",""))
    return blocks[0].get("timestamp")

def judge_mttd_mttr(
    *,
    gt_dir: str = "data/ground_truth",
    decisions_path: str = "data/decisions/decisions.jsonl",
    actions_path: str = "data/actions/enforcement_actions.jsonl",
    out_csv: str = "data/results/results_phase2.csv",
) -> str:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    decisions = _read_jsonl(decisions_path)
    actions = _read_jsonl(actions_path)

    # index decisiones por episodio (tomamos la primera o la última; aquí última)
    by_ep: Dict[int, Dict[str, Any]] = {}
    for d in decisions:
        by_ep[int(d["episode_id"])] = d

    rows = []
    for ep_id, d in sorted(by_ep.items()):
        gt = _load_ground_truth(gt_dir, ep_id)
        inj_start = _get_injection_start(gt) if gt else None

        t_detect = d.get("t_detect")
        t_block = _first_block_time(actions, ep_id)

        mttd_raw = _delta_seconds(inj_start, t_detect)
        mttr_raw = _delta_seconds(inj_start, t_block)
        anomaly: List[str] = []

        mttd = mttd_raw
        mttr = mttr_raw
        # Sanitiza el reporte principal para que no salgan tiempos negativos.
        # Se preserva el valor crudo en columnas *_raw para diagnostico.
        if mttd is not None and mttd < 0:
            anomaly.append("negative_mttd")
            mttd = 0.0
        if mttr is not None and mttr < 0:
            anomaly.append("negative_mttr")
            mttr = 0.0

        rows.append({
            "episode_id": ep_id,
            "decision": d.get("decision"),
            "inj_start": inj_start,
            "t_detect": t_detect,
            "t_block": t_block ,
            "MTTD_seconds_raw": mttd_raw,
            "MTTR_seconds_raw": mttr_raw,
            "MTTD_seconds": mttd,
            "MTTR_seconds": mttr,
            "timing_anomaly": ",".join(anomaly),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "episode_id",
            "decision",
            "inj_start",
            "t_detect",
            "t_block",
            "MTTD_seconds_raw",
            "MTTR_seconds_raw",
            "MTTD_seconds",
            "MTTR_seconds",
            "timing_anomaly",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_csv

if __name__ == "__main__":
    path = judge_mttd_mttr()
    print("Wrote:", path)
