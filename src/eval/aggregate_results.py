from __future__ import annotations

import argparse
import csv
import json
import math
import os
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _to_float(value: str) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _conf_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    out = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for row in rows:
        key = row.get("confusion") or ""
        if key in out:
            out[key] += 1
    return out


def _metric_from_confusion(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    fpr = (fp / (fp + tn)) if (fp + tn) else None
    return {"precision": precision, "recall": recall, "fpr": fpr}


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_mode(records: List[Dict[str, Any]], mode_key: str, system_key: str) -> Dict[str, Any]:
    tp_values: List[float] = []
    fp_values: List[float] = []
    tn_values: List[float] = []
    fn_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    fpr_values: List[float] = []
    episode_counts: List[float] = []

    for rec in records:
        rows = _read_csv(rec[f"{system_key}_confusion_{mode_key}"])
        counts = _conf_counts(rows)
        metrics = _metric_from_confusion(counts)
        tp_values.append(float(counts["TP"]))
        fp_values.append(float(counts["FP"]))
        tn_values.append(float(counts["TN"]))
        fn_values.append(float(counts["FN"]))
        episode_counts.append(float(len(rows)))
        if metrics["precision"] is not None:
            precision_values.append(metrics["precision"])
        if metrics["recall"] is not None:
            recall_values.append(metrics["recall"])
        if metrics["fpr"] is not None:
            fpr_values.append(metrics["fpr"])

    return {
        "system": system_key,
        "mode": mode_key,
        "n_runs": len(records),
        "episodes_per_run_mean": _fmt(_avg(episode_counts)),
        "TP_mean": _fmt(_avg(tp_values)),
        "TP_std": _fmt(_std(tp_values)),
        "FP_mean": _fmt(_avg(fp_values)),
        "FP_std": _fmt(_std(fp_values)),
        "TN_mean": _fmt(_avg(tn_values)),
        "TN_std": _fmt(_std(tn_values)),
        "FN_mean": _fmt(_avg(fn_values)),
        "FN_std": _fmt(_std(fn_values)),
        "precision_mean": _fmt(_avg(precision_values)),
        "precision_std": _fmt(_std(precision_values)),
        "recall_mean": _fmt(_avg(recall_values)),
        "recall_std": _fmt(_std(recall_values)),
        "fpr_mean": _fmt(_avg(fpr_values)),
        "fpr_std": _fmt(_std(fpr_values)),
    }


def _summarize_mttd(records: List[Dict[str, Any]], system_key: str) -> Dict[str, Any]:
    run_means: List[float] = []
    run_counts: List[float] = []
    negative_counts: List[float] = []

    for rec in records:
        rows = _read_csv(rec[f"{system_key}_phase2"])
        values = [_to_float(row.get("MTTD_seconds")) for row in rows]
        valid = [v for v in values if v is not None]
        negatives = [v for v in valid if v < 0]
        if valid:
            run_means.append(mean(valid))
        run_counts.append(float(len(valid)))
        negative_counts.append(float(len(negatives)))

    return {
        "system": system_key,
        "n_runs": len(records),
        "mttd_mean": _fmt(_avg(run_means)),
        "mttd_std": _fmt(_std(run_means)),
        "detected_episodes_mean": _fmt(_avg(run_counts)),
        "detected_episodes_std": _fmt(_std(run_counts)),
        "negative_mttd_mean": _fmt(_avg(negative_counts)),
        "negative_mttd_std": _fmt(_std(negative_counts)),
    }


def _summarize_memory_diagnosis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    attack_no_block: List[float] = []
    attack_escalate: List[float] = []
    attack_block: List[float] = []
    attack_with_hits: List[float] = []
    attack_top_fp: List[float] = []
    benign_block: List[float] = []

    for rec in records:
        rows = _read_csv(rec["blue_confusion_containment"])
        decisions = _read_jsonl(os.path.join(rec["blue_dir"], "decisions.jsonl"))
        decisions_by_ep = {int(item["episode_id"]): item for item in decisions}

        attack_rows = [row for row in rows if row.get("attack_present") == "True"]
        benign_rows = [row for row in rows if row.get("attack_present") == "False"]

        attack_no_block.append(float(sum(1 for row in attack_rows if row.get("decision") == "no_block")))
        attack_escalate.append(float(sum(1 for row in attack_rows if row.get("decision") == "escalate")))
        attack_block.append(float(sum(1 for row in attack_rows if row.get("decision") == "block_ip")))
        benign_block.append(float(sum(1 for row in benign_rows if row.get("decision") == "block_ip")))

        hit_count = 0
        fp_hit_count = 0
        for row in attack_rows:
            ep = int(row["episode_id"])
            item = decisions_by_ep.get(ep, {})
            hits = ((item.get("evidence") or {}).get("memory_hits") or [])
            if hits:
                hit_count += 1
                top_label = (hits[0].get("case") or {}).get("label")
                if top_label == "FP":
                    fp_hit_count += 1
        attack_with_hits.append(float(hit_count))
        attack_top_fp.append(float(fp_hit_count))

    return {
        "system": "blue",
        "n_runs": len(records),
        "attack_block_ip_mean": _fmt(_avg(attack_block)),
        "attack_block_ip_std": _fmt(_std(attack_block)),
        "attack_escalate_mean": _fmt(_avg(attack_escalate)),
        "attack_escalate_std": _fmt(_std(attack_escalate)),
        "attack_no_block_mean": _fmt(_avg(attack_no_block)),
        "attack_no_block_std": _fmt(_std(attack_no_block)),
        "attack_with_memory_hits_mean": _fmt(_avg(attack_with_hits)),
        "attack_with_memory_hits_std": _fmt(_std(attack_with_hits)),
        "attack_top_fp_hit_mean": _fmt(_avg(attack_top_fp)),
        "attack_top_fp_hit_std": _fmt(_std(attack_top_fp)),
        "benign_block_ip_mean": _fmt(_avg(benign_block)),
        "benign_block_ip_std": _fmt(_std(benign_block)),
    }


def _write_report(path: str, confusion_rows: List[Dict[str, Any]], mttd_rows: List[Dict[str, Any]], diagnosis_row: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("## Confusion Matrices (mean +/- std across runs)")
    lines.append("")
    for row in confusion_rows:
        lines.append(
            f"- {row['system']} / {row['mode']}: "
            f"TP {row['TP_mean']} +/- {row['TP_std']}, "
            f"FP {row['FP_mean']} +/- {row['FP_std']}, "
            f"TN {row['TN_mean']} +/- {row['TN_std']}, "
            f"FN {row['FN_mean']} +/- {row['FN_std']}, "
            f"recall {row['recall_mean']} +/- {row['recall_std']}, "
            f"n_runs {row['n_runs']}, episodes/run {row['episodes_per_run_mean']}"
        )
    lines.append("")
    lines.append("## MTTD (mean +/- std across runs)")
    lines.append("")
    for row in mttd_rows:
        lines.append(
            f"- {row['system']}: MTTD {row['mttd_mean']} +/- {row['mttd_std']}, "
            f"detected episodes/run {row['detected_episodes_mean']} +/- {row['detected_episodes_std']}, "
            f"negative MTTD/run {row['negative_mttd_mean']} +/- {row['negative_mttd_std']}"
        )
    lines.append("")
    lines.append("## Recall Diagnosis (memory run)")
    lines.append("")
    lines.append(
        "- blue: "
        f"attack_block_ip {diagnosis_row['attack_block_ip_mean']} +/- {diagnosis_row['attack_block_ip_std']}, "
        f"attack_escalate {diagnosis_row['attack_escalate_mean']} +/- {diagnosis_row['attack_escalate_std']}, "
        f"attack_no_block {diagnosis_row['attack_no_block_mean']} +/- {diagnosis_row['attack_no_block_std']}, "
        f"attack_with_memory_hits {diagnosis_row['attack_with_memory_hits_mean']} +/- {diagnosis_row['attack_with_memory_hits_std']}, "
        f"attack_top_fp_hit {diagnosis_row['attack_top_fp_hit_mean']} +/- {diagnosis_row['attack_top_fp_hit_std']}, "
        f"benign_block_ip {diagnosis_row['benign_block_ip_mean']} +/- {diagnosis_row['benign_block_ip_std']}"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: List[Dict[str, Any]] = manifest["repetitions_data"]
    experiment_dir = os.path.dirname(args.manifest)

    confusion_rows = [
        _summarize_mode(records, "containment", "baseline"),
        _summarize_mode(records, "containment", "blue"),
        _summarize_mode(records, "detection", "baseline"),
        _summarize_mode(records, "detection", "blue"),
    ]
    mttd_rows = [
        _summarize_mttd(records, "baseline"),
        _summarize_mttd(records, "blue"),
    ]
    diagnosis_row = _summarize_memory_diagnosis(records)

    _write_csv(os.path.join(experiment_dir, "summary_confusion.csv"), confusion_rows)
    _write_csv(os.path.join(experiment_dir, "summary_mttd.csv"), mttd_rows)
    _write_csv(os.path.join(experiment_dir, "diagnosis_memory.csv"), [diagnosis_row])
    _write_report(os.path.join(experiment_dir, "summary_report.md"), confusion_rows, mttd_rows, diagnosis_row)

    print("Wrote:", os.path.join(experiment_dir, "summary_confusion.csv"))
    print("Wrote:", os.path.join(experiment_dir, "summary_mttd.csv"))
    print("Wrote:", os.path.join(experiment_dir, "diagnosis_memory.csv"))
    print("Wrote:", os.path.join(experiment_dir, "summary_report.md"))


if __name__ == "__main__":
    main()
