from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_gt(gt_dir: str, episode_id: int) -> Optional[Dict[str, Any]]:
    candidates = [
        os.path.join(gt_dir, f"episode_{episode_id:03d}.json"),
        os.path.join(gt_dir, f"episode_{episode_id}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _list_gt_episode_ids(gt_dir: str) -> List[int]:
    ids: List[int] = []
    if not os.path.exists(gt_dir):
        return ids
    for name in os.listdir(gt_dir):
        if name.startswith("episode_") and name.endswith(".json"):
            stem = name[len("episode_"):-len(".json")]
            if stem.isdigit():
                ids.append(int(stem))
    return sorted(set(ids))


def _is_attack(gt: Optional[Dict[str, Any]]) -> Optional[bool]:
    if gt is None:
        return None
    if "attack_present" in gt:
        return bool(gt["attack_present"])
    # compat: si no existe el campo, asumimos que si hay scenario/techniques hay ataque
    return (gt.get("scenario_name") not in (None, "", "benign")) and bool(gt.get("technique_ids"))


def _decision_positive(decision: str, positive_decisions: List[str]) -> bool:
    return decision in set(positive_decisions)


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, is_attack: bool, predicted_positive: bool) -> None:
        if is_attack and predicted_positive:
            self.tp += 1
        elif (not is_attack) and predicted_positive:
            self.fp += 1
        elif (not is_attack) and (not predicted_positive):
            self.tn += 1
        else:
            self.fn += 1

    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else None

    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else None

    def fpr(self) -> Optional[float]:
        denom = self.fp + self.tn
        return (self.fp / denom) if denom else None


def _latest_by_episode(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_ep: Dict[int, Dict[str, Any]] = {}
    for r in records:
        by_ep[int(r["episode_id"])] = r
    return by_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/ground_truth")
    ap.add_argument("--decisions", default="data/decisions/decisions.jsonl",
                    help="Ruta a decisions.jsonl (baseline) o run decisions.jsonl.")
    ap.add_argument("--out-csv", default="data/results/results_confusion.csv")
    ap.add_argument("--positive-decisions", default="block_ip",
                    help="Comma-separated. Default: block_ip")
    args = ap.parse_args()

    positive_decisions = [x.strip() for x in args.positive_decisions.split(",") if x.strip()]

    decisions = _read_jsonl(args.decisions)
    by_ep = _latest_by_episode(decisions)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    conf = Confusion()
    mem_total = 0
    mem_top_match = 0

    rows: List[Dict[str, Any]] = []
    episode_ids = sorted(set(_list_gt_episode_ids(args.gt_dir)) | set(by_ep.keys()))
    for ep in episode_ids:
        d = by_ep.get(ep, {})
        gt = _load_gt(args.gt_dir, ep)
        attack = _is_attack(gt)
        decision = d.get("decision") or "missing"
        pred_pos = _decision_positive(decision, positive_decisions)

        # memory "identificacion" (solo si hay top hit)
        evidence = d.get("evidence") or {}
        hits = evidence.get("memory_hits") or []
        top_label = None
        top_score = None
        if hits:
            mem_total += 1
            top = hits[0]
            top_score = top.get("score")
            top_label = (top.get("case") or {}).get("label")
            if attack is not None:
                expected = "TP" if attack else "FP"
                if top_label == expected:
                    mem_top_match += 1

        outcome = None
        if attack is not None:
            conf.add(attack, pred_pos)
            if attack and pred_pos:
                outcome = "TP"
            elif (not attack) and pred_pos:
                outcome = "FP"
            elif (not attack) and (not pred_pos):
                outcome = "TN"
            else:
                outcome = "FN"

        rows.append({
            "episode_id": ep,
            "attack_present": attack,
            "decision": decision,
            "predicted_positive": pred_pos,
            "confusion": outcome,
            "top_memory_label": top_label,
            "top_memory_score": top_score,
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # resumen
    print("Wrote:", args.out_csv)
    print("TP:", conf.tp, "FP:", conf.fp, "TN:", conf.tn, "FN:", conf.fn)
    print("precision:", conf.precision(), "recall:", conf.recall(), "FPR:", conf.fpr())
    if mem_total:
        print("memory_top_label_accuracy:", mem_top_match / mem_total, f"({mem_top_match}/{mem_total})")
    else:
        print("memory_top_label_accuracy:", None, "(no memory hits)")


if __name__ == "__main__":
    main()
