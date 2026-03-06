from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ModuleNotFoundError:
    StateGraph = None  # type: ignore[assignment]
    END = None         # type: ignore[assignment]
    _HAS_LANGGRAPH = False

from src.backend_a.search_logs import search_logs
from src.tools.asset_context import get_asset_context
from src.tools.enforcement import _iso_now, block_ip
from src.blue.decision_log import append_decision
from src.memory.faiss_store import FaissMemory


_MEM_BY_DIR: dict[str, FaissMemory] = {}


def get_memory(dir_path: str) -> FaissMemory:
    mem = _MEM_BY_DIR.get(dir_path)
    if mem is None:
        mem = FaissMemory(dir_path=dir_path)
        _MEM_BY_DIR[dir_path] = mem
    return mem


def _memory_dir_from_state(state: "BlueState") -> str:
    return str(state.get("memory_dir") or os.path.join("data", "memory"))


def _ground_truth_dir_from_state(state: "BlueState") -> str:
    gt_dir = state.get("gt_dir")
    if gt_dir:
        return str(gt_dir)
    logs_dir = str(state.get("logs_dir") or "")
    return os.path.join(os.path.dirname(logs_dir), "ground_truth")


def _load_ground_truth(gt_dir: str, episode_id: int) -> Optional[Dict[str, Any]]:
    for path in (
        os.path.join(gt_dir, f"episode_{episode_id:03d}.json"),
        os.path.join(gt_dir, f"episode_{episode_id}.json"),
    ):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _memory_case_exists(mem: FaissMemory, *, text: str, label: str, episode_id: int) -> bool:
    for case in reversed(mem.cases[-300:]):
        if (
            case.get("text") == text
            and case.get("label") == label
            and case.get("source", {}).get("episode_id") == episode_id
        ):
            return True
    return False


def parse_iso_z(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": ev.get("timestamp"),
        "episode_id": ev.get("episode_id"),
        "seed": ev.get("seed"),
        "event_type": ev.get("event_type"),
        "host": ev.get("host"),
        "user": ev.get("user"),
        "src_ip": ev.get("src_ip"),
        "dst_ip": ev.get("dst_ip"),
        "action": ev.get("action"),
        "outcome": ev.get("outcome"),
        "severity": ev.get("severity", "low"),
        "process_name": ev.get("process_name"),
        "tags": ev.get("tags") or [],
    }


def build_case_text(ev: Dict[str, Any], asset_ctx: Dict[str, Any]) -> str:
    asset = asset_ctx.get("asset") or {}
    tags = " ".join(ev.get("tags") or [])
    return (
        f"event_type={ev.get('event_type')} action={ev.get('action')} outcome={ev.get('outcome')} "
        f"severity={ev.get('severity')} user={ev.get('user')} src_ip={ev.get('src_ip')} "
        f"host={ev.get('host')} role={asset.get('role')} criticality={asset.get('criticality')} tags={tags}"
    )


def _memory_summary(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "count": len(hits),
        "tp_weight": 0.0,
        "fp_weight": 0.0,
        "uncertain_weight": 0.0,
        "top_label": None,
        "top_score": 0.0,
        "consensus": "none",
    }
    if not hits:
        return summary

    top = hits[0]
    summary["top_label"] = (top.get("case") or {}).get("label")
    summary["top_score"] = float(top.get("score", 0.0))

    for hit in hits:
        label = ((hit.get("case") or {}).get("label") or "").upper()
        score = float(hit.get("score", 0.0))
        if label == "TP":
            summary["tp_weight"] += score
        elif label == "FP":
            summary["fp_weight"] += score
        elif label == "UNCERTAIN":
            summary["uncertain_weight"] += score

    max_weight = max(summary["tp_weight"], summary["fp_weight"], summary["uncertain_weight"])
    if max_weight <= 0:
        return summary
    if max_weight == summary["tp_weight"]:
        summary["consensus"] = "TP"
    elif max_weight == summary["fp_weight"]:
        summary["consensus"] = "FP"
    else:
        summary["consensus"] = "UNCERTAIN"
    return summary


def _baseline_decision(severity: str, criticality: str, signals: int) -> Dict[str, Any]:
    if severity == "high" and criticality in ("high", "medium"):
        confidence = 0.90 if signals >= 2 else 0.80
        return {
            "decision": "block_ip",
            "confidence": confidence,
            "decision_reason": "Severidad high en activo critico: contener.",
        }

    if severity == "medium" and signals >= 2:
        return {
            "decision": "block_ip",
            "confidence": 0.75,
            "decision_reason": "Evidencia suficiente (>=2 senales) con severidad medium.",
        }

    return {
        "decision": "escalate",
        "confidence": 0.60,
        "decision_reason": "Evidencia parcial: escalar (sin contener automaticamente).",
    }


class BlueState(TypedDict, total=False):
    logs_dir: str
    episode_id: int
    response_delay_sec: int
    interactive: bool
    proposed_decision: str
    final_decision: str
    gating: Dict[str, Any]
    case_text: Optional[str]
    raw_events: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    detection_event: Optional[Dict[str, Any]]
    t_detect: Optional[str]
    asset_context: Dict[str, Any]
    memory_hits: List[Dict[str, Any]]
    correlation: Dict[str, Any]
    run_id: Optional[str]
    decisions_path: Optional[str]
    actions_path: Optional[str]
    memory_dir: Optional[str]
    gt_dir: Optional[str]
    decision: str
    confidence: float
    decision_reason: str
    approved: bool
    action_result: Optional[Dict[str, Any]]


SUSPICIOUS_TAGS = ["suspicious", "lateral_like", "success_after_fail", "post_auth"]


def observe(state: BlueState) -> BlueState:
    episode_id = state["episode_id"]
    logs_dir = state["logs_dir"]

    result = search_logs(logs_dir, episode_id=episode_id, filters={"tags_any": SUSPICIOUS_TAGS}, limit=200)
    events = result["events"]

    if not events:
        fallback = search_logs(logs_dir, episode_id=episode_id, filters={"severity": "high"}, limit=200)
        events = fallback["events"]

    events.sort(key=lambda event: event.get("timestamp", ""))

    detection_event = events[0] if events else None
    t_detect = detection_event["timestamp"] if detection_event else None

    return {"raw_events": events, "detection_event": detection_event, "t_detect": t_detect}


def normalize_schema(state: BlueState) -> BlueState:
    raw = state.get("raw_events") or []
    events = [normalize_event(event) for event in raw]
    detection_event = state.get("detection_event")
    detection_event_norm = normalize_event(detection_event) if detection_event else None
    return {"events": events, "detection_event": detection_event_norm}


def enrich(state: BlueState) -> BlueState:
    detection_event = state.get("detection_event")
    if not detection_event:
        return {"asset_context": {"found": False, "notes": ["no_detection_event"]}}
    host = detection_event.get("host") or ""
    return {"asset_context": get_asset_context(host)}


def retrieve_memory(state: BlueState) -> BlueState:
    detection_event = state.get("detection_event")
    asset_context = state.get("asset_context") or {}
    if not detection_event:
        return {"memory_hits": [], "case_text": None}

    case_text = build_case_text(detection_event, asset_context)
    mem = get_memory(_memory_dir_from_state(state))
    hits = mem.search(text=case_text, k=3, threshold=0.78)
    return {
        "memory_hits": [{"score": hit.score, "case": hit.case} for hit in hits],
        "case_text": case_text,
    }


def correlate(state: BlueState) -> BlueState:
    detection_event = state.get("detection_event")
    if not detection_event:
        return {"correlation": {"signals": 0, "summary": "No detection event."}}

    logs_dir = state["logs_dir"]
    episode_id = state["episode_id"]
    src_ip = detection_event.get("src_ip")
    t_detect = detection_event.get("timestamp")
    signals = 1
    window: Dict[str, Any] = {}

    if src_ip and t_detect:
        t0 = parse_iso_z(t_detect)
        start = iso_z(t0 - timedelta(minutes=2))
        end = iso_z(t0 + timedelta(minutes=2))
        result = search_logs(
            logs_dir,
            episode_id=episode_id,
            start=start,
            end=end,
            filters={"src_ip": src_ip},
            limit=200,
            agg={"type": "count"},
        )
        count = result.get("aggregation", {}).get("count", 0)
        signals = 2 if count >= 5 else 1
        window = {"start": start, "end": end, "src_ip_count": count}

    correlation = {
        "primary": {
            "event_type": detection_event.get("event_type"),
            "action": detection_event.get("action"),
            "severity": detection_event.get("severity"),
            "tags": detection_event.get("tags"),
            "src_ip": src_ip,
            "host": detection_event.get("host"),
        },
        "window": window,
        "signals": signals,
    }
    return {"correlation": correlation}


def decide(state: BlueState) -> BlueState:
    detection_event = state.get("detection_event")
    if not detection_event:
        return {
            "decision": "no_block",
            "confidence": 0.20,
            "decision_reason": "Sin evento de deteccion: no se actua.",
        }

    asset = (state.get("asset_context") or {}).get("asset") or {}
    criticality = asset.get("criticality", "low")
    tags = detection_event.get("tags") or []
    severity = detection_event.get("severity", "low")
    signals = (state.get("correlation") or {}).get("signals", 1)

    allowlisted = ("allowlisted_user" in tags) or ("service_account" in tags)
    if allowlisted:
        return {
            "decision": "no_block",
            "confidence": 0.95,
            "decision_reason": "Allowlisted/service_account: evitar falso positivo.",
        }

    base = _baseline_decision(severity, criticality, signals)
    hits = state.get("memory_hits") or []
    if not hits:
        return base

    memory = _memory_summary(hits)
    top_score = float(memory["top_score"])
    tp_weight = float(memory["tp_weight"])
    fp_weight = float(memory["fp_weight"])
    base_decision = str(base["decision"])
    base_confidence = float(base["confidence"])

    if memory["consensus"] == "TP" and top_score >= 0.90 and tp_weight >= (fp_weight + 0.10):
        if base_decision == "escalate":
            return {
                "decision": "block_ip",
                "confidence": max(base_confidence, 0.85),
                "decision_reason": f"Memoria respalda TP fuerte (score={top_score:.2f}): contener.",
            }
        return {
            "decision": base_decision,
            "confidence": max(base_confidence, 0.88),
            "decision_reason": f"{base['decision_reason']} | Memoria consistente con TP (score={top_score:.2f}).",
        }

    if memory["consensus"] == "FP" and top_score >= 0.86 and fp_weight >= (tp_weight + 0.15):
        if base_decision == "block_ip":
            if severity == "high" and criticality in ("high", "medium") and signals >= 2:
                return {
                    "decision": "block_ip",
                    "confidence": min(base_confidence, 0.82),
                    "decision_reason": f"{base['decision_reason']} | Memoria sugiere FP, pero la evidencia actual sigue siendo fuerte.",
                }
            return {
                "decision": "escalate",
                "confidence": 0.72,
                "decision_reason": f"Memoria sugiere FP similar (score={top_score:.2f}): pedir mayor evidencia antes de contener.",
            }
        return {
            "decision": "no_block",
            "confidence": 0.82,
            "decision_reason": f"Memoria sugiere FP similar (score={top_score:.2f}) y la evidencia actual es debil.",
        }

    if memory["count"] >= 2:
        return {
            "decision": base_decision,
            "confidence": max(0.55, base_confidence - 0.08),
            "decision_reason": f"{base['decision_reason']} | Memoria mixta o insuficiente; se prioriza la heuristica base.",
        }

    return base


def act(state: BlueState) -> BlueState:
    proposed = state.get("decision", "no_block")
    detection_event = state.get("detection_event") or {}
    t_detect = state.get("t_detect")
    delay = int(state.get("response_delay_sec", 30))
    interactive = bool(state.get("interactive", True))

    gating = {"prompted": False, "approved": True, "reason": None}
    approved = True
    final_decision = proposed
    action_result = None

    if proposed == "block_ip":
        confidence = float(state.get("confidence", 0.0))
        if confidence < 0.80 and interactive:
            gating["prompted"] = True
            answer = input(f"[GATING] Bloquear IP? confidence={confidence:.2f} (y/n): ").strip().lower()
            approved = (answer == "y")
            gating["approved"] = approved
            if not approved:
                gating["reason"] = "human_rejected"
                final_decision = "escalate"

        if approved:
            src_ip = detection_event.get("src_ip")
            if src_ip and t_detect:
                t0 = parse_iso_z(t_detect)
                t_block = iso_z(t0 + timedelta(seconds=delay))
                action_result = block_ip(
                    src_ip,
                    1800,
                    episode_id=state["episode_id"],
                    reason=f"blue_agent: {state.get('decision_reason')}",
                    action_time=t_block,
                    out_path=state.get("actions_path"),
                    run_id=state.get("run_id"),
                )
            else:
                approved = False
                gating["approved"] = False
                gating["reason"] = "missing_ip_or_time"
                final_decision = "escalate"

    return {
        "proposed_decision": proposed,
        "final_decision": final_decision,
        "approved": approved,
        "gating": gating,
        "action_result": action_result,
    }


def log(state: BlueState) -> BlueState:
    episode_id = state["episode_id"]
    run_id = state.get("run_id")
    decisions_path = state.get("decisions_path") or os.path.join("data", "runs", run_id or "unknown_run", "decisions.jsonl")

    proposed = state.get("proposed_decision", state.get("decision", "no_block"))
    final = state.get("final_decision", proposed)
    gating = state.get("gating") or {}
    approved = bool(gating.get("approved", state.get("approved", True)))

    reason = state.get("decision_reason", "")
    if proposed == "block_ip" and final != "block_ip":
        reason = (reason + " | Bloqueo propuesto pero rechazado por gating.").strip()

    case_text = state.get("case_text")
    if case_text:
        gt = _load_ground_truth(_ground_truth_dir_from_state(state), episode_id)
        attack_present = bool((gt or {}).get("attack_present"))
        signals = int((state.get("correlation") or {}).get("signals", 1))
        severity = str(((state.get("detection_event") or {}).get("severity")) or "low")
        tags = list(((state.get("detection_event") or {}).get("tags")) or [])

        label: Optional[str] = None
        learned_decision: Optional[str] = None
        learned_reason: Optional[str] = None
        learned_tags: List[str] = []

        if gt is not None:
            if attack_present:
                label = "TP"
                learned_decision = "block_ip" if (severity == "high" or signals >= 2) else "escalate"
                learned_reason = "ground_truth_attack_present"
                learned_tags = ["ground_truth_feedback", "attack_present"]
            else:
                label = "FP"
                learned_decision = "no_block" if signals <= 1 else "escalate"
                learned_reason = "ground_truth_benign"
                learned_tags = ["ground_truth_feedback", "benign"]
                if "service_account" in tags or "allowlisted_user" in tags:
                    learned_tags.append("allowlist_pattern")

        if gating.get("prompted") is True and label is None:
            label = "TP" if approved else "FP"
            learned_decision = final
            learned_reason = f"gating_feedback: approved={approved}"
            learned_tags = ["gating_feedback"]

        if label and learned_decision and learned_reason:
            mem = get_memory(_memory_dir_from_state(state))
            if not _memory_case_exists(mem, text=case_text, label=label, episode_id=episode_id):
                mem.add_case(
                    text=case_text,
                    label=label,
                    decision=learned_decision,
                    reason=learned_reason,
                    tags=learned_tags,
                    confidence=state.get("confidence"),
                    source={"episode_id": episode_id, "run_id": run_id},
                )

    evidence = {
        "run_id": run_id,
        "proposed_decision": proposed,
        "final_decision": final,
        "gating": gating,
        "approved": approved,
        "detection_event": state.get("detection_event"),
        "asset_context": state.get("asset_context"),
        "memory_hits": state.get("memory_hits"),
        "correlation": state.get("correlation"),
        "action_result": state.get("action_result"),
    }

    append_decision(
        episode_id=episode_id,
        decision=final,
        t_detect=state.get("t_detect"),
        evidence=evidence,
        reason=reason,
        run_id=run_id,
        out_path=decisions_path,
        timestamp=_iso_now(),
    )

    return {}


def build_blue_graph():
    if not _HAS_LANGGRAPH:
        raise RuntimeError("langgraph no esta instalado; usa run_blue_episode(state) como fallback.")

    graph = StateGraph(BlueState)
    graph.add_node("observe", observe)
    graph.add_node("normalize_schema", normalize_schema)
    graph.add_node("enrich", enrich)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("correlate", correlate)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_node("log", log)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "normalize_schema")
    graph.add_edge("normalize_schema", "enrich")
    graph.add_edge("enrich", "retrieve_memory")
    graph.add_edge("retrieve_memory", "correlate")
    graph.add_edge("correlate", "decide")
    graph.add_edge("decide", "act")
    graph.add_edge("act", "log")
    graph.add_edge("log", END)

    return graph.compile()


def run_blue_episode(state: BlueState) -> BlueState:
    current: BlueState = dict(state)
    for fn in (observe, normalize_schema, enrich, retrieve_memory, correlate, decide, act, log):
        out = fn(current) or {}
        current.update(out)
    return current
