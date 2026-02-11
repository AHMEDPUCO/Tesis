from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
import os

from src.backend_a.search_logs import search_logs
from src.tools.asset_context import get_asset_context
from src.tools.enforcement import _iso_now, block_ip
from src.blue.decision_log import append_decision

##Faiss memory importado pero no integrado aún (ver retrieve_memory):
from src.memory.faiss_store import FaissMemory



#-----------------faiss_memory-----------------"
# arriba del todo
_MEM_BY_DIR: dict[str, FaissMemory] = {}

def get_memory(dir_path: str) -> FaissMemory:
    m = _MEM_BY_DIR.get(dir_path)
    if m is None:
        m = FaissMemory(dir_path=dir_path)
        _MEM_BY_DIR[dir_path] = m
    return m

# Cache global para evitar recargar el índice en cada llamada (si decides usarlo)"
_MEM: FaissMemory | None = None
def get_memory() -> FaissMemory:
    global _MEM
    if _MEM is None:
        _MEM = FaissMemory()
    return _MEM
def _memory_case_exists(mem, *, text: str, label: str, episode_id: int) -> bool:
    for c in reversed(mem.cases[-300:]):  # mira los últimos 300
        if c.get("text") == text and c.get("label") == label and c.get("source", {}).get("episode_id") == episode_id:
            return True
    return False
# --------- helpers ---------

def parse_iso_z(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def normalize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """normalize_schema: asegura llaves consistentes (aunque falten en el raw)."""
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
    a = asset_ctx.get("asset") or {}
    tags = " ".join(ev.get("tags") or [])
    return (
        f"event_type={ev.get('event_type')} action={ev.get('action')} outcome={ev.get('outcome')} "
        f"severity={ev.get('severity')} user={ev.get('user')} src_ip={ev.get('src_ip')} "
        f"host={ev.get('host')} role={a.get('role')} criticality={a.get('criticality')} tags={tags}"
    )


# --------- state ---------
class BlueState(TypedDict, total=False):
    logs_dir: str
    episode_id: int
    response_delay_sec: int
    interactive: bool
    proposed_decision: str
    final_decision: str
    gating: Dict[str, Any]
    case_text: Optional[str]
    # outputs intermedios
    raw_events: List[Dict[str, Any]]
    events: List[Dict[str, Any]]           # normalized
    detection_event: Optional[Dict[str, Any]]
    t_detect: Optional[str]
    asset_context: Dict[str, Any]
    memory_hits: List[Dict[str, Any]]
    correlation: Dict[str, Any]
    #run context
    run_id: Optional[str]
    decisions_path: Optional[str]
    actions_path: Optional[str]
    ##########
    # decisión/acción
    decision: str                          # block_ip | no_block | escalate
    confidence: float
    decision_reason: str
    approved: bool
    action_result: Optional[Dict[str, Any]]


# --------- nodes ---------
SUSPICIOUS_TAGS = ["suspicious", "lateral_like", "burst", "success_after_fail", "post_auth"]

def observe(state: BlueState) -> BlueState:
    ep = state["episode_id"]
    logs_dir = state["logs_dir"]

    # 1) busca señales por tags sospechosos
    r = search_logs(logs_dir, episode_id=ep, filters={"tags_any": SUSPICIOUS_TAGS}, limit=200)
    events = r["events"]

    # fallback: si no hay tags, busca severidad high (más agresivo)
    if not events:
        r2 = search_logs(logs_dir, episode_id=ep, filters={"severity": "high"}, limit=200)
        events = r2["events"]

    # ordenar por timestamp
    events.sort(key=lambda e: e.get("timestamp", ""))

    det = events[0] if events else None
    t_detect = det["timestamp"] if det else None

    return {"raw_events": events, "detection_event": det, "t_detect": t_detect}

def normalize_schema(state: BlueState) -> BlueState:
    raw = state.get("raw_events") or []
    norm = [normalize_event(e) for e in raw]
    det = state.get("detection_event")
    det_norm = normalize_event(det) if det else None
    return {"events": norm, "detection_event": det_norm}

def enrich(state: BlueState) -> BlueState:
    det = state.get("detection_event")
    if not det:
        return {"asset_context": {"found": False, "notes": ["no_detection_event"]}}
    host = det.get("host") or ""
    ctx = get_asset_context(host)
    return {"asset_context": ctx}

def retrieve_memory(state: BlueState) -> BlueState:
    det = state.get("detection_event")
    asset_ctx = state.get("asset_context") or {}

    if not det:
        return {"memory_hits": [], "case_text": None}

    case_text = build_case_text(det, asset_ctx)

    mem = get_memory()
    hits = mem.search(text=case_text, k=3, threshold=0.78)

    hits_dict = [
        {"score": h.score, "case": h.case}
        for h in hits
    ]

    return {"memory_hits": hits_dict, "case_text": case_text}


def correlate(state: BlueState) -> BlueState:
    det = state.get("detection_event")
    if not det:
        return {"correlation": {"signals": 0, "summary": "No detection event."}}

    # Evidencia adicional: cuántos eventos comparten src_ip cerca de t_detect
    logs_dir = state["logs_dir"]
    ep = state["episode_id"]
    src_ip = det.get("src_ip")
    t_detect = det.get("timestamp")
    signals = 1

    window = {}
    if src_ip and t_detect:
        t0 = parse_iso_z(t_detect)
        start = iso_z(t0 - timedelta(minutes=2))
        end = iso_z(t0 + timedelta(minutes=2))

        rr = search_logs(
            logs_dir,
            episode_id=ep,
            start=start,
            end=end,
            filters={"src_ip": src_ip},
            limit=200,
            agg={"type": "count"},
        )
        cnt = rr.get("aggregation", {}).get("count", 0)
        signals = 2 if cnt >= 5 else 1
        window = {"start": start, "end": end, "src_ip_count": cnt}

    summary = {
        "primary": {
            "event_type": det.get("event_type"),
            "action": det.get("action"),
            "severity": det.get("severity"),
            "tags": det.get("tags"),
            "src_ip": src_ip,
            "host": det.get("host"),
        },
        "window": window,
        "signals": signals,
    }
    return {"correlation": summary}

def decide(state: BlueState) -> BlueState:
    det = state.get("detection_event")
    if not det:
        return {
            "decision": "no_block",
            "confidence": 0.20,
            "decision_reason": "Sin evento de detección: no se actúa."
        }

    asset = (state.get("asset_context") or {}).get("asset") or {}
    crit = asset.get("criticality", "low")

    tags = det.get("tags") or []
    sev = det.get("severity", "low")
    signals = (state.get("correlation") or {}).get("signals", 1)

    # 1) Allowlist override (prioridad máxima)
    allowlisted = ("allowlisted_user" in tags) or ("service_account" in tags)
    if allowlisted:
        return {
            "decision": "no_block",
            "confidence": 0.95,
            "decision_reason": "Allowlisted/service_account: evitar falso positivo."
        }

    # 2) Memoria FAISS override (segundo nivel)
    hits = state.get("memory_hits") or []
    if hits:
        top = hits[0]
        top_label = (top.get("case") or {}).get("label")
        top_score = float(top.get("score", 0.0))

        # Si memoria sugiere FP con alta similitud, NO bloquear
        if top_label == "FP" and top_score >= 0.82:
            return {
                "decision": "no_block",
                "confidence": 0.88,
                "decision_reason": f"Memoria sugiere FP similar (score={top_score:.2f}): evitar bloqueo."
            }

        # Si memoria sugiere TP muy fuerte, favorecer bloqueo (pero con prudencia)
        if top_label == "TP" and top_score >= 0.90:
            # bloquea con alta confianza pero deja que gating actúe si tu umbral lo exige
            return {
                "decision": "block_ip",
                "confidence": 0.90,
                "decision_reason": f"Memoria sugiere TP similar (score={top_score:.2f}): contener."
            }

        # Si el match existe pero no es fuerte, se deja que la heurística decida
        # (no hacemos return aquí)

    # 3) Heurística v0 (baseline)
    if sev == "high" and (crit in ("high", "medium")):
        conf = 0.90 if signals >= 2 else 0.80
        return {
            "decision": "block_ip",
            "confidence": conf,
            "decision_reason": "Severidad high en activo crítico: contener."
        }

    if sev == "medium" and signals >= 2:
        return {
            "decision": "block_ip",
            "confidence": 0.75,
            "decision_reason": "Evidencia suficiente (>=2 señales) con severidad medium."
        }

    return {
        "decision": "escalate",
        "confidence": 0.60,
        "decision_reason": "Evidencia parcial: escalar (sin contener automáticamente)."
    }

def act(state: BlueState) -> BlueState:
    proposed = state.get("decision", "no_block")
    det = state.get("detection_event") or {}
    t_detect = state.get("t_detect")
    delay = int(state.get("response_delay_sec", 30))
    interactive = bool(state.get("interactive", True))

    gating = {"prompted": False, "approved": True, "reason": None}
    approved = True
    final_decision = proposed
    action_result = None

    if proposed == "block_ip":
        conf = float(state.get("confidence", 0.0))

        # gating simple: si conf < 0.8, pide aprobación
        if conf < 0.80 and interactive:
            gating["prompted"] = True
            ans = input(f"[GATING] Bloquear IP? confidence={conf:.2f} (y/n): ").strip().lower()
            approved = (ans == "y")
            gating["approved"] = approved
            if not approved:
                gating["reason"] = "human_rejected"
                final_decision = "escalate"   # bloque propuesto, pero NO ejecutado

        # ejecuta acción solo si aprobado
        if approved:
            src_ip = det.get("src_ip")
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
    ep = state["episode_id"]

    # === Run context (para no mezclar corridas) ===
    run_id = state.get("run_id")
    decisions_path = state.get("decisions_path") or os.path.join("data", "runs", run_id or "unknown_run", "decisions.jsonl")
    actions_path = state.get("actions_path") or os.path.join("data", "runs", run_id or "unknown_run", "enforcement_actions.jsonl")
    # === decisiones ===
    proposed = state.get("proposed_decision", state.get("decision", "no_block"))
    final = state.get("final_decision", proposed)

    gating = state.get("gating") or {}
    approved = bool(gating.get("approved", state.get("approved", True)))

    reason = state.get("decision_reason", "")
    if proposed == "block_ip" and final != "block_ip":
        reason = (reason + " | Bloqueo propuesto pero rechazado por gating.").strip()

    # === FAISS memory: guardar SOLO si hubo gating ===
    case_text = state.get("case_text")
    if case_text and gating.get("prompted") is True:
        label = "TP" if approved else "FP"
        mem = get_memory()  # o get_memory(), según tu implementación

        # dedupe: no metas el mismo caso una y otra vez
        if not _memory_case_exists(mem, text=case_text, label=label, episode_id=ep):
            mem.add_case(
                text=case_text,
                label=label,
                decision=final,
                reason=f"gating_feedback: approved={approved}",
                tags=["gating_feedback"],
                source={"episode_id": ep, "run_id": run_id},
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

    # IMPORTANTe: append_decision debe poder escribir al path del run
    append_decision(
        episode_id=ep,
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
    g = StateGraph(BlueState)
    g.add_node("observe", observe)
    g.add_node("normalize_schema", normalize_schema)
    g.add_node("enrich", enrich)
    g.add_node("retrieve_memory", retrieve_memory)
    g.add_node("correlate", correlate)
    g.add_node("decide", decide)
    g.add_node("act", act)
    g.add_node("log", log)

    g.set_entry_point("observe")
    g.add_edge("observe", "normalize_schema")
    g.add_edge("normalize_schema", "enrich")
    g.add_edge("enrich", "retrieve_memory")
    g.add_edge("retrieve_memory", "correlate")
    g.add_edge("correlate", "decide")
    g.add_edge("decide", "act")
    g.add_edge("act", "log")
    g.add_edge("log", END)

    return g.compile()
## PARA EJECUTAR BLUE AGENT python -m src.blue.run_blue_agent --episode-id 2
