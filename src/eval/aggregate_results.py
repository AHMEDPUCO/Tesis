from __future__ import annotations

import argparse
import csv
import json
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


def _safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def _decision_distribution(rows: List[Dict[str, str]]) -> Dict[str, float]:
    total = float(len(rows))
    attack_rows = [row for row in rows if row.get("attack_present") == "True"]
    benign_rows = [row for row in rows if row.get("attack_present") == "False"]

    block = float(sum(1 for row in rows if row.get("decision") == "block_ip"))
    esc = float(sum(1 for row in rows if row.get("decision") == "escalate"))
    noblock = float(sum(1 for row in rows if row.get("decision") == "no_block"))
    attack_noblock = float(sum(1 for row in attack_rows if row.get("decision") == "no_block"))
    attack_action = float(sum(1 for row in attack_rows if row.get("decision") in ("block_ip", "escalate")))
    benign_action = float(sum(1 for row in benign_rows if row.get("decision") in ("block_ip", "escalate")))

    return {
        "total_episodes": total,
        "attack_episodes": float(len(attack_rows)),
        "benign_episodes": float(len(benign_rows)),
        "block_count": block,
        "escalate_count": esc,
        "no_block_count": noblock,
        "action_rate_total": _safe_div(block + esc, total),
        "action_rate_attack": _safe_div(attack_action, float(len(attack_rows))),
        "action_rate_benign": _safe_div(benign_action, float(len(benign_rows))),
        "attack_no_block_rate": _safe_div(attack_noblock, float(len(attack_rows))),
    }


def _episode_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _episode_backend(decision_item: Dict[str, Any]) -> str:
    evidence = decision_item.get("evidence") or {}
    search_tool = evidence.get("search_tool") or {}
    backend = search_tool.get("backend")
    if backend:
        return str(backend)
    for call in (search_tool.get("calls") or []):
        b = call.get("backend")
        if b:
            return str(b)
    return "unknown"


def _schema_source(decision_item: Dict[str, Any]) -> str:
    evidence = decision_item.get("evidence") or {}
    schema_mapping = evidence.get("schema_mapping") or {}
    source = schema_mapping.get("source")
    return str(source) if source else "none"


def _schema_mapping_meta(decision_item: Dict[str, Any]) -> Dict[str, Any]:
    evidence = decision_item.get("evidence") or {}
    schema_mapping = evidence.get("schema_mapping") or {}
    return {
        "source": str(schema_mapping.get("source") or "none"),
        "cache_hit": bool(schema_mapping.get("cache_hit", False)),
        "llm_called": bool(schema_mapping.get("llm_called", False)),
    }


def _subset_confusion_by_backend(
    rows: List[Dict[str, str]],
    backend_by_episode: Dict[int, str],
    backend: str,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        ep = _episode_int(row.get("episode_id"))
        if ep is None:
            continue
        if backend_by_episode.get(ep) == backend:
            out.append(row)
    return out


def _phase_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def _summarize_swap_phase_per_run(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        decisions = _read_jsonl(os.path.join(rec["blue_dir"], "decisions.jsonl"))
        cont_rows = _read_csv(rec["blue_confusion_containment"])
        det_rows = _read_csv(rec["blue_confusion_detection"])
        phase_rows = _read_csv(rec["blue_phase2"])

        backend_by_episode: Dict[int, str] = {}
        first_ep_by_backend: Dict[str, int] = {}
        for item in decisions:
            ep = _episode_int(item.get("episode_id"))
            if ep is None:
                continue
            backend = _episode_backend(item)
            backend_by_episode[ep] = backend
            if backend not in first_ep_by_backend:
                first_ep_by_backend[backend] = ep

        ordered_backends = sorted(first_ep_by_backend.keys(), key=lambda b: first_ep_by_backend[b])
        phase1 = ordered_backends[0] if ordered_backends else ""
        phase2 = ordered_backends[1] if len(ordered_backends) > 1 else ""
        has_swap = len(ordered_backends) > 1

        cont_p1 = _subset_confusion_by_backend(cont_rows, backend_by_episode, phase1) if phase1 else []
        cont_p2 = _subset_confusion_by_backend(cont_rows, backend_by_episode, phase2) if phase2 else []
        det_p1 = _subset_confusion_by_backend(det_rows, backend_by_episode, phase1) if phase1 else []
        det_p2 = _subset_confusion_by_backend(det_rows, backend_by_episode, phase2) if phase2 else []

        cont_p1_metrics = _metric_from_confusion(_conf_counts(cont_p1)) if cont_p1 else {"recall": None, "fpr": None}
        cont_p2_metrics = _metric_from_confusion(_conf_counts(cont_p2)) if cont_p2 else {"recall": None, "fpr": None}
        det_p1_metrics = _metric_from_confusion(_conf_counts(det_p1)) if det_p1 else {"recall": None}
        det_p2_metrics = _metric_from_confusion(_conf_counts(det_p2)) if det_p2 else {"recall": None}

        mttd_p1: List[float] = []
        mttd_p2: List[float] = []
        mttr_p1: List[float] = []
        mttr_p2: List[float] = []
        for row in phase_rows:
            ep = _episode_int(row.get("episode_id"))
            if ep is None:
                continue
            backend = backend_by_episode.get(ep)
            mttd = _to_float(row.get("MTTD_seconds"))
            mttr = _to_float(row.get("MTTR_seconds"))
            if backend == phase1:
                if mttd is not None:
                    mttd_p1.append(mttd)
                if mttr is not None:
                    mttr_p1.append(mttr)
            elif backend == phase2:
                if mttd is not None:
                    mttd_p2.append(mttd)
                if mttr is not None:
                    mttr_p2.append(mttr)

        rec_p1 = cont_p1_metrics.get("recall")
        rec_p2 = cont_p2_metrics.get("recall")
        det_rec_p1 = det_p1_metrics.get("recall")
        det_rec_p2 = det_p2_metrics.get("recall")
        fpr_p1 = cont_p1_metrics.get("fpr")
        fpr_p2 = cont_p2_metrics.get("fpr")
        mttd_mean_p1 = _phase_mean(mttd_p1)
        mttd_mean_p2 = _phase_mean(mttd_p2)
        mttr_mean_p1 = _phase_mean(mttr_p1)
        mttr_mean_p2 = _phase_mean(mttr_p2)

        out.append(
            {
                "run_id": rec["blue_run_id"],
                "repetition": rec["repetition"],
                "has_swap": has_swap,
                "phase1_backend": phase1,
                "phase1_episodes": len(cont_p1),
                "phase1_recall_containment": _fmt(rec_p1),
                "phase1_recall_detection": _fmt(det_rec_p1),
                "phase1_fpr_containment": _fmt(fpr_p1),
                "phase1_mttd_mean": _fmt(mttd_mean_p1),
                "phase1_mttr_mean": _fmt(mttr_mean_p1),
                "phase2_backend": phase2,
                "phase2_episodes": len(cont_p2),
                "phase2_recall_containment": _fmt(rec_p2),
                "phase2_recall_detection": _fmt(det_rec_p2),
                "phase2_fpr_containment": _fmt(fpr_p2),
                "phase2_mttd_mean": _fmt(mttd_mean_p2),
                "phase2_mttr_mean": _fmt(mttr_mean_p2),
                "delta_recall_containment_p2_minus_p1": _fmt(
                    None if rec_p1 is None or rec_p2 is None else (rec_p2 - rec_p1)
                ),
                "delta_recall_detection_p2_minus_p1": _fmt(
                    None if det_rec_p1 is None or det_rec_p2 is None else (det_rec_p2 - det_rec_p1)
                ),
                "delta_fpr_containment_p2_minus_p1": _fmt(
                    None if fpr_p1 is None or fpr_p2 is None else (fpr_p2 - fpr_p1)
                ),
                "delta_mttd_p2_minus_p1": _fmt(
                    None if mttd_mean_p1 is None or mttd_mean_p2 is None else (mttd_mean_p2 - mttd_mean_p1)
                ),
                "delta_mttr_p2_minus_p1": _fmt(
                    None if mttr_mean_p1 is None or mttr_mean_p2 is None else (mttr_mean_p2 - mttr_mean_p1)
                ),
            }
        )
    return out


def _summarize_swap_phase_summary(per_run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    swap_rows = [r for r in per_run_rows if str(r.get("has_swap")).lower() == "true"]
    if not swap_rows:
        return {"n_runs": len(per_run_rows), "runs_with_swap": 0}

    def vals(key: str) -> List[float]:
        v: List[float] = []
        for row in swap_rows:
            x = _to_float(str(row.get(key, "")))
            if x is not None:
                v.append(x)
        return v

    return {
        "n_runs": len(per_run_rows),
        "runs_with_swap": len(swap_rows),
        "phase1_recall_containment_mean": _fmt(_avg(vals("phase1_recall_containment"))),
        "phase1_recall_containment_std": _fmt(_std(vals("phase1_recall_containment"))),
        "phase2_recall_containment_mean": _fmt(_avg(vals("phase2_recall_containment"))),
        "phase2_recall_containment_std": _fmt(_std(vals("phase2_recall_containment"))),
        "delta_recall_containment_mean": _fmt(_avg(vals("delta_recall_containment_p2_minus_p1"))),
        "delta_recall_containment_std": _fmt(_std(vals("delta_recall_containment_p2_minus_p1"))),
        "phase1_mttd_mean": _fmt(_avg(vals("phase1_mttd_mean"))),
        "phase1_mttd_std": _fmt(_std(vals("phase1_mttd_mean"))),
        "phase2_mttd_mean": _fmt(_avg(vals("phase2_mttd_mean"))),
        "phase2_mttd_std": _fmt(_std(vals("phase2_mttd_mean"))),
        "delta_mttd_mean": _fmt(_avg(vals("delta_mttd_p2_minus_p1"))),
        "delta_mttd_std": _fmt(_std(vals("delta_mttd_p2_minus_p1"))),
        "phase1_mttr_mean": _fmt(_avg(vals("phase1_mttr_mean"))),
        "phase1_mttr_std": _fmt(_std(vals("phase1_mttr_mean"))),
        "phase2_mttr_mean": _fmt(_avg(vals("phase2_mttr_mean"))),
        "phase2_mttr_std": _fmt(_std(vals("phase2_mttr_mean"))),
        "delta_mttr_mean": _fmt(_avg(vals("delta_mttr_p2_minus_p1"))),
        "delta_mttr_std": _fmt(_std(vals("delta_mttr_p2_minus_p1"))),
    }


def _counts_and_metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    counts = _conf_counts(rows)
    metrics = _metric_from_confusion(counts)
    return {
        "episodes": len(rows),
        "TP": counts["TP"],
        "FP": counts["FP"],
        "TN": counts["TN"],
        "FN": counts["FN"],
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "fpr": metrics.get("fpr"),
    }


def _summarize_schema_fallback_per_run(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        decisions = _read_jsonl(os.path.join(rec["blue_dir"], "decisions.jsonl"))
        cont_rows = _read_csv(rec["blue_confusion_containment"])

        source_by_episode: Dict[int, str] = {}
        for item in decisions:
            ep = _episode_int(item.get("episode_id"))
            if ep is None:
                continue
            source_by_episode[ep] = _schema_source(item)

        fallback_rows: List[Dict[str, str]] = []
        non_fallback_rows: List[Dict[str, str]] = []
        fallback_episodes = 0
        for row in cont_rows:
            ep = _episode_int(row.get("episode_id"))
            if ep is None:
                continue
            source = source_by_episode.get(ep, "none")
            if source.startswith("fallback"):
                fallback_rows.append(row)
                fallback_episodes += 1
            else:
                non_fallback_rows.append(row)

        fb = _counts_and_metrics(fallback_rows)
        nf = _counts_and_metrics(non_fallback_rows)
        total = len(cont_rows)
        fallback_rate = (fallback_episodes / total) if total else None

        out.append(
            {
                "run_id": rec["blue_run_id"],
                "repetition": rec["repetition"],
                "episodes_total": total,
                "fallback_episodes": fallback_episodes,
                "fallback_rate": _fmt(fallback_rate),
                "fallback_recall": _fmt(fb["recall"]),
                "fallback_fpr": _fmt(fb["fpr"]),
                "fallback_fp": fb["FP"],
                "fallback_tp": fb["TP"],
                "non_fallback_episodes": nf["episodes"],
                "non_fallback_recall": _fmt(nf["recall"]),
                "non_fallback_fpr": _fmt(nf["fpr"]),
                "non_fallback_fp": nf["FP"],
                "non_fallback_tp": nf["TP"],
                "delta_recall_nonfb_minus_fb": _fmt(
                    None if nf["recall"] is None or fb["recall"] is None else (nf["recall"] - fb["recall"])
                ),
                "delta_fpr_nonfb_minus_fb": _fmt(
                    None if nf["fpr"] is None or fb["fpr"] is None else (nf["fpr"] - fb["fpr"])
                ),
            }
        )
    return out


def _summarize_schema_fallback_summary(per_run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for row in per_run_rows:
            v = _to_float(str(row.get(key, "")))
            if v is not None:
                out.append(v)
        return out

    return {
        "n_runs": len(per_run_rows),
        "fallback_rate_mean": _fmt(_avg(vals("fallback_rate"))),
        "fallback_rate_std": _fmt(_std(vals("fallback_rate"))),
        "fallback_recall_mean": _fmt(_avg(vals("fallback_recall"))),
        "fallback_recall_std": _fmt(_std(vals("fallback_recall"))),
        "non_fallback_recall_mean": _fmt(_avg(vals("non_fallback_recall"))),
        "non_fallback_recall_std": _fmt(_std(vals("non_fallback_recall"))),
        "fallback_fpr_mean": _fmt(_avg(vals("fallback_fpr"))),
        "fallback_fpr_std": _fmt(_std(vals("fallback_fpr"))),
        "non_fallback_fpr_mean": _fmt(_avg(vals("non_fallback_fpr"))),
        "non_fallback_fpr_std": _fmt(_std(vals("non_fallback_fpr"))),
        "delta_recall_nonfb_minus_fb_mean": _fmt(_avg(vals("delta_recall_nonfb_minus_fb"))),
        "delta_recall_nonfb_minus_fb_std": _fmt(_std(vals("delta_recall_nonfb_minus_fb"))),
    }


def _summarize_memory_coverage_per_run(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        decisions = _read_jsonl(os.path.join(rec["blue_dir"], "decisions.jsonl"))
        cont_rows = _read_csv(rec["blue_confusion_containment"])
        conf_by_ep: Dict[int, str] = {}
        for row in cont_rows:
            ep = _episode_int(row.get("episode_id"))
            if ep is not None:
                conf_by_ep[ep] = str(row.get("confusion") or "")

        total = len(decisions)
        hit_count = 0
        override_count = 0
        influence_count = 0
        reinforce_count = 0
        promote_count = 0
        downgrade_count = 0
        override_conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        no_override_conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        for item in decisions:
            ep = _episode_int(item.get("episode_id"))
            if ep is None:
                continue
            evidence = item.get("evidence") or {}
            hits = evidence.get("memory_hits") or []
            has_hit = len(hits) > 0
            if has_hit:
                hit_count += 1

            trace = evidence.get("decision_trace") or {}
            base = str(trace.get("base_decision") or evidence.get("proposed_decision") or item.get("decision") or "")
            final = str(evidence.get("final_decision") or item.get("decision") or "")
            changed = base != final
            is_override = has_hit and changed
            if is_override:
                override_count += 1
            memory_rule = str(trace.get("memory_rule_applied") or "")
            memory_influenced = has_hit and memory_rule not in ("", "no_memory_hits", "memory_observed_no_override")
            if memory_influenced:
                influence_count += 1
                if "reinforce" in memory_rule or "keep_block" in memory_rule:
                    reinforce_count += 1
                elif "promote" in memory_rule:
                    promote_count += 1
                elif "downgrade" in memory_rule or "suppress" in memory_rule:
                    downgrade_count += 1

            conf = conf_by_ep.get(ep)
            if conf in override_conf:
                if is_override:
                    override_conf[conf] += 1
                else:
                    no_override_conf[conf] += 1

        coverage = (hit_count / total) if total else None
        override_rate = (override_count / total) if total else None
        override_given_hit = (override_count / hit_count) if hit_count else None
        influence_rate = (influence_count / total) if total else None
        influence_given_hit = (influence_count / hit_count) if hit_count else None

        out.append(
            {
                "run_id": rec["blue_run_id"],
                "repetition": rec["repetition"],
                "episodes_total": total,
                "memory_hit_episodes": hit_count,
                "memory_coverage_rate": _fmt(coverage),
                "memory_override_episodes": override_count,
                "memory_override_rate": _fmt(override_rate),
                "memory_override_given_hit_rate": _fmt(override_given_hit),
                "memory_influence_episodes": influence_count,
                "memory_influence_rate": _fmt(influence_rate),
                "memory_influence_given_hit_rate": _fmt(influence_given_hit),
                "memory_reinforce_episodes": reinforce_count,
                "memory_promote_episodes": promote_count,
                "memory_downgrade_episodes": downgrade_count,
                "override_TP": override_conf["TP"],
                "override_FP": override_conf["FP"],
                "override_TN": override_conf["TN"],
                "override_FN": override_conf["FN"],
                "no_override_TP": no_override_conf["TP"],
                "no_override_FP": no_override_conf["FP"],
                "no_override_TN": no_override_conf["TN"],
                "no_override_FN": no_override_conf["FN"],
            }
        )
    return out


def _summarize_memory_coverage_summary(per_run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for row in per_run_rows:
            v = _to_float(str(row.get(key, "")))
            if v is not None:
                out.append(v)
        return out

    return {
        "n_runs": len(per_run_rows),
        "memory_coverage_rate_mean": _fmt(_avg(vals("memory_coverage_rate"))),
        "memory_coverage_rate_std": _fmt(_std(vals("memory_coverage_rate"))),
        "memory_override_rate_mean": _fmt(_avg(vals("memory_override_rate"))),
        "memory_override_rate_std": _fmt(_std(vals("memory_override_rate"))),
        "memory_override_given_hit_rate_mean": _fmt(_avg(vals("memory_override_given_hit_rate"))),
        "memory_override_given_hit_rate_std": _fmt(_std(vals("memory_override_given_hit_rate"))),
        "memory_influence_rate_mean": _fmt(_avg(vals("memory_influence_rate"))),
        "memory_influence_rate_std": _fmt(_std(vals("memory_influence_rate"))),
        "memory_influence_given_hit_rate_mean": _fmt(_avg(vals("memory_influence_given_hit_rate"))),
        "memory_influence_given_hit_rate_std": _fmt(_std(vals("memory_influence_given_hit_rate"))),
        "memory_promote_episodes_mean": _fmt(_avg(vals("memory_promote_episodes"))),
        "memory_promote_episodes_std": _fmt(_std(vals("memory_promote_episodes"))),
        "memory_downgrade_episodes_mean": _fmt(_avg(vals("memory_downgrade_episodes"))),
        "memory_downgrade_episodes_std": _fmt(_std(vals("memory_downgrade_episodes"))),
    }


def _summarize_schema_mapper_usage_per_run(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        decisions = _read_jsonl(os.path.join(rec["blue_dir"], "decisions.jsonl"))
        total = len(decisions)
        llm_called = 0
        cache_hits = 0
        gemini_source = 0
        ollama_source = 0
        fallback_source = 0
        none_source = 0
        for item in decisions:
            meta = _schema_mapping_meta(item)
            source = meta["source"]
            if meta["llm_called"]:
                llm_called += 1
            if meta["cache_hit"]:
                cache_hits += 1
            if source == "gemini":
                gemini_source += 1
            elif source == "ollama":
                ollama_source += 1
            elif source.startswith("fallback"):
                fallback_source += 1
            elif source == "none":
                none_source += 1
        out.append(
            {
                "run_id": rec["blue_run_id"],
                "repetition": rec["repetition"],
                "episodes_total": total,
                "llm_called_episodes": llm_called,
                "llm_call_rate": _fmt(_safe_div(float(llm_called), float(total))),
                "cache_hit_episodes": cache_hits,
                "cache_hit_rate": _fmt(_safe_div(float(cache_hits), float(total))),
                "gemini_source_episodes": gemini_source,
                "gemini_source_rate": _fmt(_safe_div(float(gemini_source), float(total))),
                "ollama_source_episodes": ollama_source,
                "ollama_source_rate": _fmt(_safe_div(float(ollama_source), float(total))),
                "fallback_source_episodes": fallback_source,
                "fallback_source_rate": _fmt(_safe_div(float(fallback_source), float(total))),
                "none_source_episodes": none_source,
                "none_source_rate": _fmt(_safe_div(float(none_source), float(total))),
            }
        )
    return out


def _summarize_schema_mapper_usage_summary(per_run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for row in per_run_rows:
            v = _to_float(str(row.get(key, "")))
            if v is not None:
                out.append(v)
        return out

    return {
        "n_runs": len(per_run_rows),
        "llm_call_rate_mean": _fmt(_avg(vals("llm_call_rate"))),
        "llm_call_rate_std": _fmt(_std(vals("llm_call_rate"))),
        "cache_hit_rate_mean": _fmt(_avg(vals("cache_hit_rate"))),
        "cache_hit_rate_std": _fmt(_std(vals("cache_hit_rate"))),
        "gemini_source_rate_mean": _fmt(_avg(vals("gemini_source_rate"))),
        "gemini_source_rate_std": _fmt(_std(vals("gemini_source_rate"))),
        "ollama_source_rate_mean": _fmt(_avg(vals("ollama_source_rate"))),
        "ollama_source_rate_std": _fmt(_std(vals("ollama_source_rate"))),
        "fallback_source_rate_mean": _fmt(_avg(vals("fallback_source_rate"))),
        "fallback_source_rate_std": _fmt(_std(vals("fallback_source_rate"))),
        "none_source_rate_mean": _fmt(_avg(vals("none_source_rate"))),
        "none_source_rate_std": _fmt(_std(vals("none_source_rate"))),
    }


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


def _summarize_tradeoff(records: List[Dict[str, Any]], system_key: str) -> Dict[str, Any]:
    fp_values: List[float] = []
    fn_values: List[float] = []
    recall_cont_values: List[float] = []
    recall_det_values: List[float] = []
    precision_cont_values: List[float] = []

    action_rate_total_values: List[float] = []
    action_rate_attack_values: List[float] = []
    action_rate_benign_values: List[float] = []
    attack_no_block_rate_values: List[float] = []
    block_count_values: List[float] = []
    escalate_count_values: List[float] = []
    no_block_count_values: List[float] = []
    total_episode_values: List[float] = []

    for rec in records:
        cont_rows = _read_csv(rec[f"{system_key}_confusion_containment"])
        det_rows = _read_csv(rec[f"{system_key}_confusion_detection"])
        cont_counts = _conf_counts(cont_rows)
        det_counts = _conf_counts(det_rows)
        cont_metrics = _metric_from_confusion(cont_counts)
        det_metrics = _metric_from_confusion(det_counts)
        dist = _decision_distribution(cont_rows)

        fp_values.append(float(cont_counts["FP"]))
        fn_values.append(float(cont_counts["FN"]))
        if cont_metrics["precision"] is not None:
            precision_cont_values.append(float(cont_metrics["precision"]))
        if cont_metrics["recall"] is not None:
            recall_cont_values.append(float(cont_metrics["recall"]))
        if det_metrics["recall"] is not None:
            recall_det_values.append(float(det_metrics["recall"]))

        if dist["action_rate_total"] is not None:
            action_rate_total_values.append(float(dist["action_rate_total"]))
        if dist["action_rate_attack"] is not None:
            action_rate_attack_values.append(float(dist["action_rate_attack"]))
        if dist["action_rate_benign"] is not None:
            action_rate_benign_values.append(float(dist["action_rate_benign"]))
        if dist["attack_no_block_rate"] is not None:
            attack_no_block_rate_values.append(float(dist["attack_no_block_rate"]))

        block_count_values.append(float(dist["block_count"]))
        escalate_count_values.append(float(dist["escalate_count"]))
        no_block_count_values.append(float(dist["no_block_count"]))
        total_episode_values.append(float(dist["total_episodes"]))

    return {
        "system": system_key,
        "n_runs": len(records),
        "episodes_per_run_mean": _fmt(_avg(total_episode_values)),
        "FP_mean": _fmt(_avg(fp_values)),
        "FP_std": _fmt(_std(fp_values)),
        "FN_mean": _fmt(_avg(fn_values)),
        "FN_std": _fmt(_std(fn_values)),
        "precision_containment_mean": _fmt(_avg(precision_cont_values)),
        "precision_containment_std": _fmt(_std(precision_cont_values)),
        "recall_containment_mean": _fmt(_avg(recall_cont_values)),
        "recall_containment_std": _fmt(_std(recall_cont_values)),
        "recall_detection_mean": _fmt(_avg(recall_det_values)),
        "recall_detection_std": _fmt(_std(recall_det_values)),
        "action_rate_total_mean": _fmt(_avg(action_rate_total_values)),
        "action_rate_total_std": _fmt(_std(action_rate_total_values)),
        "action_rate_attack_mean": _fmt(_avg(action_rate_attack_values)),
        "action_rate_attack_std": _fmt(_std(action_rate_attack_values)),
        "action_rate_benign_mean": _fmt(_avg(action_rate_benign_values)),
        "action_rate_benign_std": _fmt(_std(action_rate_benign_values)),
        "attack_no_block_rate_mean": _fmt(_avg(attack_no_block_rate_values)),
        "attack_no_block_rate_std": _fmt(_std(attack_no_block_rate_values)),
        "block_count_mean": _fmt(_avg(block_count_values)),
        "block_count_std": _fmt(_std(block_count_values)),
        "escalate_count_mean": _fmt(_avg(escalate_count_values)),
        "escalate_count_std": _fmt(_std(escalate_count_values)),
        "no_block_count_mean": _fmt(_avg(no_block_count_values)),
        "no_block_count_std": _fmt(_std(no_block_count_values)),
    }


def _summarize_mttd(records: List[Dict[str, Any]], system_key: str) -> Dict[str, Any]:
    run_means: List[float] = []
    run_counts: List[float] = []
    negative_counts: List[float] = []

    for rec in records:
        rows = _read_csv(rec[f"{system_key}_phase2"])
        # Usa columna saneada para promedio y columna raw para diagnostico
        # de anomalias negativas si existe.
        values = [_to_float(row.get("MTTD_seconds")) for row in rows]
        raw_values = [
            _to_float(row.get("MTTD_seconds_raw"))
            if row.get("MTTD_seconds_raw") not in ("", None)
            else _to_float(row.get("MTTD_seconds"))
            for row in rows
        ]
        valid = [v for v in values if v is not None]
        negatives = [v for v in raw_values if v is not None and v < 0]
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


def _summarize_latency_breakdown(records: List[Dict[str, Any]], system_key: str) -> Dict[str, Any]:
    stages = [
        "observe",
        "normalize_schema",
        "enrich",
        "retrieve_memory",
        "correlate",
        "decide",
        "act",
        "log",
    ]

    run_pipeline_means: List[float] = []
    run_with_timing_counts: List[float] = []
    stage_run_means: Dict[str, List[float]] = {stage: [] for stage in stages}

    for rec in records:
        decisions_path = os.path.join(rec[f"{system_key}_dir"], "decisions.jsonl")
        decisions = _read_jsonl(decisions_path)

        pipeline_vals: List[float] = []
        per_stage_vals: Dict[str, List[float]] = {stage: [] for stage in stages}
        with_timing = 0

        for item in decisions:
            timing = ((item.get("evidence") or {}).get("timing") or {})
            if not timing:
                continue
            with_timing += 1

            pipeline_ms = _to_float(str(timing.get("pipeline_duration_ms")))
            if pipeline_ms is not None:
                pipeline_vals.append(pipeline_ms)

            stage_map = timing.get("stages") or {}
            for stage in stages:
                duration = _to_float(str(((stage_map.get(stage) or {}).get("duration_ms"))))
                if duration is not None:
                    per_stage_vals[stage].append(duration)

        run_with_timing_counts.append(float(with_timing))
        if pipeline_vals:
            run_pipeline_means.append(mean(pipeline_vals))
        for stage in stages:
            if per_stage_vals[stage]:
                stage_run_means[stage].append(mean(per_stage_vals[stage]))

    row: Dict[str, Any] = {
        "system": system_key,
        "n_runs": len(records),
        "decisions_with_timing_mean": _fmt(_avg(run_with_timing_counts)),
        "decisions_with_timing_std": _fmt(_std(run_with_timing_counts)),
        "pipeline_duration_ms_mean": _fmt(_avg(run_pipeline_means)),
        "pipeline_duration_ms_std": _fmt(_std(run_pipeline_means)),
    }
    for stage in stages:
        row[f"{stage}_ms_mean"] = _fmt(_avg(stage_run_means[stage]))
        row[f"{stage}_ms_std"] = _fmt(_std(stage_run_means[stage]))
    return row


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


def _write_report(
    path: str,
    confusion_rows: List[Dict[str, Any]],
    mttd_rows: List[Dict[str, Any]],
    diagnosis_row: Dict[str, Any],
    tradeoff_rows: List[Dict[str, Any]],
    latency_rows: List[Dict[str, Any]],
    swap_phase_summary: Dict[str, Any],
    schema_fallback_summary: Dict[str, Any],
    memory_coverage_summary: Dict[str, Any],
    schema_mapper_usage_summary: Dict[str, Any],
) -> None:
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
    lines.append("## FP/FN vs Action Tradeoff (mean +/- std across runs)")
    lines.append("")
    for row in tradeoff_rows:
        lines.append(
            f"- {row['system']}: "
            f"FP {row['FP_mean']} +/- {row['FP_std']}, "
            f"FN {row['FN_mean']} +/- {row['FN_std']}, "
            f"recall_cont {row['recall_containment_mean']} +/- {row['recall_containment_std']}, "
            f"recall_det {row['recall_detection_mean']} +/- {row['recall_detection_std']}, "
            f"action_total {row['action_rate_total_mean']} +/- {row['action_rate_total_std']}, "
            f"action_attack {row['action_rate_attack_mean']} +/- {row['action_rate_attack_std']}, "
            f"attack_no_block {row['attack_no_block_rate_mean']} +/- {row['attack_no_block_rate_std']}"
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
    lines.append("## Latency Breakdown (mean +/- std across runs)")
    lines.append("")
    for row in latency_rows:
        lines.append(
            f"- {row['system']}: pipeline_ms {row['pipeline_duration_ms_mean']} +/- {row['pipeline_duration_ms_std']}, "
            f"timed_decisions/run {row['decisions_with_timing_mean']} +/- {row['decisions_with_timing_std']}, "
            f"observe {row['observe_ms_mean']} +/- {row['observe_ms_std']}, "
            f"retrieve_memory {row['retrieve_memory_ms_mean']} +/- {row['retrieve_memory_ms_std']}, "
            f"correlate {row['correlate_ms_mean']} +/- {row['correlate_ms_std']}, "
            f"decide {row['decide_ms_mean']} +/- {row['decide_ms_std']}, "
            f"act {row['act_ms_mean']} +/- {row['act_ms_std']}, "
            f"log {row['log_ms_mean']} +/- {row['log_ms_std']}"
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
    lines.append("")
    lines.append("## Swap Phase Metrics (blue)")
    lines.append("")
    lines.append(
        f"- runs_with_swap: {swap_phase_summary.get('runs_with_swap', 0)} / {swap_phase_summary.get('n_runs', 0)}, "
        f"phase1_recall_cont {swap_phase_summary.get('phase1_recall_containment_mean', '')} +/- {swap_phase_summary.get('phase1_recall_containment_std', '')}, "
        f"phase2_recall_cont {swap_phase_summary.get('phase2_recall_containment_mean', '')} +/- {swap_phase_summary.get('phase2_recall_containment_std', '')}, "
        f"delta_recall_cont {swap_phase_summary.get('delta_recall_containment_mean', '')} +/- {swap_phase_summary.get('delta_recall_containment_std', '')}, "
        f"phase1_mttd {swap_phase_summary.get('phase1_mttd_mean', '')} +/- {swap_phase_summary.get('phase1_mttd_std', '')}, "
        f"phase2_mttd {swap_phase_summary.get('phase2_mttd_mean', '')} +/- {swap_phase_summary.get('phase2_mttd_std', '')}, "
        f"delta_mttd {swap_phase_summary.get('delta_mttd_mean', '')} +/- {swap_phase_summary.get('delta_mttd_std', '')}"
    )
    lines.append("")
    lines.append("## Schema Fallback Impact (blue)")
    lines.append("")
    lines.append(
        f"- fallback_rate {schema_fallback_summary.get('fallback_rate_mean', '')} +/- {schema_fallback_summary.get('fallback_rate_std', '')}, "
        f"fallback_recall {schema_fallback_summary.get('fallback_recall_mean', '')} +/- {schema_fallback_summary.get('fallback_recall_std', '')}, "
        f"non_fallback_recall {schema_fallback_summary.get('non_fallback_recall_mean', '')} +/- {schema_fallback_summary.get('non_fallback_recall_std', '')}, "
        f"fallback_fpr {schema_fallback_summary.get('fallback_fpr_mean', '')} +/- {schema_fallback_summary.get('fallback_fpr_std', '')}, "
        f"non_fallback_fpr {schema_fallback_summary.get('non_fallback_fpr_mean', '')} +/- {schema_fallback_summary.get('non_fallback_fpr_std', '')}, "
        f"delta_recall(nonfb-fb) {schema_fallback_summary.get('delta_recall_nonfb_minus_fb_mean', '')} +/- {schema_fallback_summary.get('delta_recall_nonfb_minus_fb_std', '')}"
    )
    lines.append("")
    lines.append("## Memory Coverage & Override (blue)")
    lines.append("")
    lines.append(
        f"- memory_coverage_rate {memory_coverage_summary.get('memory_coverage_rate_mean', '')} +/- {memory_coverage_summary.get('memory_coverage_rate_std', '')}, "
        f"memory_override_rate {memory_coverage_summary.get('memory_override_rate_mean', '')} +/- {memory_coverage_summary.get('memory_override_rate_std', '')}, "
        f"override_given_hit_rate {memory_coverage_summary.get('memory_override_given_hit_rate_mean', '')} +/- {memory_coverage_summary.get('memory_override_given_hit_rate_std', '')}, "
        f"memory_influence_rate {memory_coverage_summary.get('memory_influence_rate_mean', '')} +/- {memory_coverage_summary.get('memory_influence_rate_std', '')}, "
        f"influence_given_hit_rate {memory_coverage_summary.get('memory_influence_given_hit_rate_mean', '')} +/- {memory_coverage_summary.get('memory_influence_given_hit_rate_std', '')}, "
        f"promote_episodes/run {memory_coverage_summary.get('memory_promote_episodes_mean', '')} +/- {memory_coverage_summary.get('memory_promote_episodes_std', '')}, "
        f"downgrade_episodes/run {memory_coverage_summary.get('memory_downgrade_episodes_mean', '')} +/- {memory_coverage_summary.get('memory_downgrade_episodes_std', '')}"
    )
    lines.append("")
    lines.append("## Schema Mapper Usage (blue)")
    lines.append("")
    lines.append(
        f"- llm_call_rate {schema_mapper_usage_summary.get('llm_call_rate_mean', '')} +/- {schema_mapper_usage_summary.get('llm_call_rate_std', '')}, "
        f"cache_hit_rate {schema_mapper_usage_summary.get('cache_hit_rate_mean', '')} +/- {schema_mapper_usage_summary.get('cache_hit_rate_std', '')}, "
        f"gemini_source_rate {schema_mapper_usage_summary.get('gemini_source_rate_mean', '')} +/- {schema_mapper_usage_summary.get('gemini_source_rate_std', '')}, "
        f"ollama_source_rate {schema_mapper_usage_summary.get('ollama_source_rate_mean', '')} +/- {schema_mapper_usage_summary.get('ollama_source_rate_std', '')}, "
        f"fallback_source_rate {schema_mapper_usage_summary.get('fallback_source_rate_mean', '')} +/- {schema_mapper_usage_summary.get('fallback_source_rate_std', '')}, "
        f"none_source_rate {schema_mapper_usage_summary.get('none_source_rate_mean', '')} +/- {schema_mapper_usage_summary.get('none_source_rate_std', '')}"
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
    latency_rows = [
        _summarize_latency_breakdown(records, "baseline"),
        _summarize_latency_breakdown(records, "blue"),
    ]
    tradeoff_rows = [
        _summarize_tradeoff(records, "baseline"),
        _summarize_tradeoff(records, "blue"),
    ]
    diagnosis_row = _summarize_memory_diagnosis(records)
    swap_phase_per_run = _summarize_swap_phase_per_run(records)
    swap_phase_summary = _summarize_swap_phase_summary(swap_phase_per_run)
    schema_fallback_per_run = _summarize_schema_fallback_per_run(records)
    schema_fallback_summary = _summarize_schema_fallback_summary(schema_fallback_per_run)
    memory_coverage_per_run = _summarize_memory_coverage_per_run(records)
    memory_coverage_summary = _summarize_memory_coverage_summary(memory_coverage_per_run)
    schema_mapper_usage_per_run = _summarize_schema_mapper_usage_per_run(records)
    schema_mapper_usage_summary = _summarize_schema_mapper_usage_summary(schema_mapper_usage_per_run)

    _write_csv(os.path.join(experiment_dir, "summary_confusion.csv"), confusion_rows)
    _write_csv(os.path.join(experiment_dir, "summary_mttd.csv"), mttd_rows)
    _write_csv(os.path.join(experiment_dir, "latency_breakdown.csv"), latency_rows)
    _write_csv(os.path.join(experiment_dir, "tradeoff_fp_fn_action.csv"), tradeoff_rows)
    _write_csv(os.path.join(experiment_dir, "diagnosis_memory.csv"), [diagnosis_row])
    _write_csv(os.path.join(experiment_dir, "swap_phase_metrics.csv"), swap_phase_per_run)
    _write_csv(os.path.join(experiment_dir, "swap_phase_summary.csv"), [swap_phase_summary])
    _write_csv(os.path.join(experiment_dir, "schema_fallback_impact.csv"), schema_fallback_per_run)
    _write_csv(os.path.join(experiment_dir, "schema_fallback_summary.csv"), [schema_fallback_summary])
    _write_csv(os.path.join(experiment_dir, "memory_coverage_override.csv"), memory_coverage_per_run)
    _write_csv(os.path.join(experiment_dir, "memory_coverage_summary.csv"), [memory_coverage_summary])
    _write_csv(os.path.join(experiment_dir, "schema_mapper_usage.csv"), schema_mapper_usage_per_run)
    _write_csv(os.path.join(experiment_dir, "schema_mapper_usage_summary.csv"), [schema_mapper_usage_summary])
    _write_report(
        os.path.join(experiment_dir, "summary_report.md"),
        confusion_rows,
        mttd_rows,
        diagnosis_row,
        tradeoff_rows,
        latency_rows,
        swap_phase_summary,
        schema_fallback_summary,
        memory_coverage_summary,
        schema_mapper_usage_summary,
    )

    print("Wrote:", os.path.join(experiment_dir, "summary_confusion.csv"))
    print("Wrote:", os.path.join(experiment_dir, "summary_mttd.csv"))
    print("Wrote:", os.path.join(experiment_dir, "latency_breakdown.csv"))
    print("Wrote:", os.path.join(experiment_dir, "tradeoff_fp_fn_action.csv"))
    print("Wrote:", os.path.join(experiment_dir, "diagnosis_memory.csv"))
    print("Wrote:", os.path.join(experiment_dir, "swap_phase_metrics.csv"))
    print("Wrote:", os.path.join(experiment_dir, "swap_phase_summary.csv"))
    print("Wrote:", os.path.join(experiment_dir, "schema_fallback_impact.csv"))
    print("Wrote:", os.path.join(experiment_dir, "schema_fallback_summary.csv"))
    print("Wrote:", os.path.join(experiment_dir, "memory_coverage_override.csv"))
    print("Wrote:", os.path.join(experiment_dir, "memory_coverage_summary.csv"))
    print("Wrote:", os.path.join(experiment_dir, "schema_mapper_usage.csv"))
    print("Wrote:", os.path.join(experiment_dir, "schema_mapper_usage_summary.csv"))
    print("Wrote:", os.path.join(experiment_dir, "summary_report.md"))


if __name__ == "__main__":
    main()
