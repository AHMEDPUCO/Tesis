from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_by_episode(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    for row in records:
        latest[int(row["episode_id"])] = row
    return latest


def _fmt_ratio(num: int, den: int) -> str:
    return f"{(num / den):.4f}" if den else ""


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _avg(values)
    if mean is None:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _fmt_float(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else ""


def _analyze_run(gt_dir: str, decisions_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    decisions = _latest_by_episode(_read_jsonl(decisions_path))
    per_episode_rows: List[Dict[str, Any]] = []

    for name in sorted(os.listdir(gt_dir)):
        if not name.startswith("episode_") or not name.endswith(".json"):
            continue
        gt = _read_json(os.path.join(gt_dir, name))
        if not gt:
            continue
        attack_present = bool(gt.get("attack_present"))
        benign_pattern_type = str(gt.get("benign_pattern_type") or "")
        scenario_name = str(gt.get("scenario_name") or "")
        if attack_present:
            continue
        if benign_pattern_type != "recurrent" and not scenario_name.startswith("benign_recurrent_"):
            continue

        episode_id = int(gt.get("episode_id") or 0)
        record = decisions.get(episode_id) or {}
        evidence = record.get("evidence") or {}
        hits = list(evidence.get("memory_hits") or [])
        decision_trace = evidence.get("decision_trace") or {}
        top_label = None
        top_score = None
        if hits:
            top_label = ((hits[0].get("case") or {}).get("label"))
            top_score = hits[0].get("score")

        final_decision = str(record.get("decision") or "missing")
        proposed_decision = str(evidence.get("proposed_decision") or "")
        base_decision = str(decision_trace.get("base_decision") or "")
        memory_rule = str(decision_trace.get("memory_rule_applied") or "")
        memory_consensus = str(decision_trace.get("memory_consensus") or "")
        memory_hit = bool(hits)

        per_episode_rows.append(
            {
                "episode_id": episode_id,
                "scenario_name": scenario_name,
                "benign_pattern_profile": gt.get("benign_pattern_profile") or "",
                "decision": final_decision,
                "proposed_decision": proposed_decision,
                "base_decision": base_decision,
                "memory_hit": memory_hit,
                "memory_consensus": memory_consensus,
                "memory_rule_applied": memory_rule,
                "top_memory_label": top_label,
                "top_memory_score": top_score,
            }
        )

    total = len(per_episode_rows)
    no_block = sum(1 for row in per_episode_rows if row["decision"] == "no_block")
    escalate = sum(1 for row in per_episode_rows if row["decision"] == "escalate")
    block_ip = sum(1 for row in per_episode_rows if row["decision"] == "block_ip")
    memory_hits = sum(1 for row in per_episode_rows if row["memory_hit"])
    fp_consensus = sum(1 for row in per_episode_rows if row["memory_consensus"] == "FP")
    suppressed_rules = {
        "fp_strong_suppress_action",
        "fp_recurrent_suppress_action",
        "fp_recurrent_exact_suppress_action",
    }
    downgraded_rules = {
        "fp_strong_downgrade_to_escalate",
        "fp_recurrent_downgrade_to_escalate",
    }
    suppressed = sum(1 for row in per_episode_rows if row["memory_rule_applied"] in suppressed_rules)
    downgraded = sum(1 for row in per_episode_rows if row["memory_rule_applied"] in downgraded_rules)

    return per_episode_rows, {
        "episodes_total": total,
        "no_block_episodes": no_block,
        "no_block_rate": _fmt_ratio(no_block, total),
        "escalate_episodes": escalate,
        "escalate_rate": _fmt_ratio(escalate, total),
        "block_ip_episodes": block_ip,
        "block_ip_rate": _fmt_ratio(block_ip, total),
        "memory_hit_episodes": memory_hits,
        "memory_hit_rate": _fmt_ratio(memory_hits, total),
        "memory_fp_consensus_episodes": fp_consensus,
        "memory_fp_consensus_rate": _fmt_ratio(fp_consensus, total),
        "memory_suppressed_episodes": suppressed,
        "memory_suppressed_rate": _fmt_ratio(suppressed, total),
        "memory_downgraded_episodes": downgraded,
        "memory_downgraded_rate": _fmt_ratio(downgraded, total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary-csv", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary_path = args.out_summary_csv or os.path.splitext(args.out_csv)[0] + "_summary.csv"

    if args.manifest:
        manifest = _read_json(args.manifest) or {}
        rep_rows: List[Dict[str, Any]] = []
        for rep in manifest.get("repetitions_data") or []:
            gt_dir = rep.get("gt_dir")
            decisions_path = os.path.join(str(rep.get("blue_dir") or ""), "decisions.jsonl")
            if not gt_dir or not os.path.exists(str(gt_dir)) or not os.path.exists(decisions_path):
                continue
            _, summary = _analyze_run(str(gt_dir), decisions_path)
            rep_rows.append(
                {
                    "repetition": rep.get("repetition"),
                    **summary,
                }
            )

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            if rep_rows:
                writer = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
                writer.writeheader()
                for row in rep_rows:
                    writer.writerow(row)
            else:
                writer = csv.writer(f)
                writer.writerow(["repetition"])

        rate_keys = [
            "no_block_rate",
            "escalate_rate",
            "block_ip_rate",
            "memory_hit_rate",
            "memory_fp_consensus_rate",
            "memory_suppressed_rate",
            "memory_downgraded_rate",
        ]
        summary_row = {"runs": len(rep_rows)}
        for key in rate_keys:
            values = [_to_float(row.get(key)) for row in rep_rows]
            float_values = [value for value in values if value is not None]
            summary_row[f"{key}_mean"] = _fmt_float(_avg(float_values))
            summary_row[f"{key}_std"] = _fmt_float(_std(float_values))

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)

        print("Wrote:", args.out_csv)
        print("Wrote:", summary_path)
        print(summary_row)
        return

    if not args.gt_dir or not args.decisions:
        raise SystemExit("Provide --manifest or both --gt-dir and --decisions.")

    per_episode_rows, summary_row = _analyze_run(args.gt_dir, args.decisions)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode_id",
            "scenario_name",
            "benign_pattern_profile",
            "decision",
            "proposed_decision",
            "base_decision",
            "memory_hit",
            "memory_consensus",
            "memory_rule_applied",
            "top_memory_label",
            "top_memory_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_episode_rows:
            writer.writerow(row)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print("Wrote:", args.out_csv)
    print("Wrote:", summary_path)
    print(summary_row)


if __name__ == "__main__":
    main()
