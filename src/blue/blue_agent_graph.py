from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ModuleNotFoundError:
    StateGraph = None  # type: ignore[assignment]
    END = None         # type: ignore[assignment]
    _HAS_LANGGRAPH = False

from src.backend_a.search_logs import search_logs as search_logs_backend_a
from src.backend_b.search_logs import search_logs as search_logs_backend_b
from src.tools.asset_context import get_asset_context
from src.tools.enforcement import _iso_now, block_ip
from src.blue.decision_log import append_decision
from src.blue.schema_mapper import DynamicSchemaMapper, FALLBACK_ALIASES
from src.memory.faiss_store import FaissMemory
from src.mcp import LocalMCPClient


_MEM_BY_DIR: dict[str, FaissMemory] = {}
_MAPPER_BY_CACHE: dict[str, DynamicSchemaMapper] = {}
_MCP_BY_KEY: dict[str, LocalMCPClient] = {}

BACKEND_B_ALIASES_FULL: Dict[str, str] = {
    "timestamp": "when_utc|ts_epoch_ms|tstamp|time_obs",
    "episode_id": "case_ref|case_num|incident_case|ep_ref",
    "seed": "rnd|rand_seed|seed_id|rnd_id",
    "event_type": "evt_kind|evt_type_name|kind|cat",
    "host": "asset_ref|asset_name|node|host_ref",
    "user": "actor_id|principal|account|usr_ref",
    "src_ip": "origin_addr|src_addr|ip_from|src",
    "dst_ip": "target_addr|dst_addr|ip_to|dst",
    "action": "op_name|op|verb|op_name_v2",
    "outcome": "result_state|ok|result_code|state_text",
    "severity": "risk_code|sev_level|priority|risk",
    "process_name": "proc_image|proc|image|proc_path|proc_meta.image",
    "tags": "labels_v2|tag_blob|labels|tagset",
}

# Minimal contract for drift-hard experiments: deliberately incomplete core mapping.
BACKEND_B_ALIASES_MINIMAL: Dict[str, str] = {
    "timestamp": "when_utc|ts_epoch_ms|tstamp|time_obs",
    "event_type": "evt_kind|evt_type_name|kind|cat",
    "tags": "labels_v2|tag_blob|labels|tagset",
}


def get_memory(dir_path: str) -> FaissMemory:
    mem = _MEM_BY_DIR.get(dir_path)
    if mem is None:
        mem = FaissMemory(dir_path=dir_path)
        _MEM_BY_DIR[dir_path] = mem
    return mem


def _memory_dir_from_state(state: "BlueState") -> str:
    return str(state.get("memory_dir") or os.path.join("data", "memory"))


MEMORY_SEARCH_THRESHOLD = 0.62
MEMORY_TP_STRONG_THRESHOLD = 0.82
MEMORY_FP_STRONG_THRESHOLD = 0.86
MEMORY_TP_MARGIN = 0.02
MEMORY_FP_MARGIN = 0.20
MEMORY_TP_PROMOTION_THRESHOLD = 0.74


def _schema_map_cache_path_from_state(state: "BlueState") -> str:
    explicit = state.get("schema_map_cache_path")
    if explicit:
        return str(explicit)
    scope = str(state.get("schema_cache_scope") or "run")
    if scope == "persistent":
        return os.path.join("data", "memory", "schema_map_cache.json")
    return os.path.join(_memory_dir_from_state(state), "schema_map_cache.json")


def _schema_map_shared_cache_path_from_state(state: "BlueState") -> str:
    llm_provider = str(state.get("llm_provider") or "gemini")
    if llm_provider == "ollama":
        model = str(state.get("ollama_model") or os.getenv("OLLAMA_MODEL") or "qwen3:8b")
    else:
        model = str(state.get("gemini_model") or "gemini-1.5-flash")
    backend_b_alias_mode = str(state.get("backend_b_alias_mode") or "full")
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "default"
    return os.path.join(
        "data",
        "experiments",
        "_shared_schema_cache",
        f"schema_map_cache_shared_{llm_provider}_{model_slug}_{backend_b_alias_mode}.json",
    )


def get_schema_mapper(state: "BlueState") -> DynamicSchemaMapper:
    mode = str(state.get("schema_mapper_mode") or "static")
    enabled = mode == "dynamic"
    llm_provider = str(state.get("llm_provider") or "gemini")
    model = str(state.get("gemini_model") or "gemini-1.5-flash")
    raw_min_conf = state.get("schema_map_min_confidence")
    min_confidence = 0.75 if raw_min_conf is None else float(raw_min_conf)
    ollama_url = str(state.get("ollama_url") or os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434")
    ollama_model = str(state.get("ollama_model") or os.getenv("OLLAMA_MODEL") or "qwen3:8b")
    llm_timeout_sec = float(state.get("llm_timeout_sec") or os.getenv("SCHEMA_MAP_TIMEOUT_SEC") or 8.0)
    cache_path = _schema_map_cache_path_from_state(state)
    shared_cache_path = _schema_map_shared_cache_path_from_state(state)
    api_key = str(state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY") or "")
    mapper_key = (
        f"{cache_path}|{enabled}|{llm_provider}|{model}|{min_confidence:.3f}|"
        f"{ollama_url}|{ollama_model}|{llm_timeout_sec:.3f}|{bool(api_key)}|{shared_cache_path}"
    )

    mapper = _MAPPER_BY_CACHE.get(mapper_key)
    if mapper is None:
        mapper = DynamicSchemaMapper(
            enabled=enabled,
            cache_path=cache_path,
            api_key=api_key or None,
            provider=llm_provider,
            model=model,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            min_confidence=min_confidence,
            llm_timeout_sec=llm_timeout_sec,
            shared_cache_path=shared_cache_path,
        )
        _MAPPER_BY_CACHE[mapper_key] = mapper
    return mapper


def get_mcp_client(state: "BlueState") -> LocalMCPClient:
    mcp_tool = str(state.get("mcp_tool") or "search_logs")
    backend_b_alias_mode = str(state.get("backend_b_alias_mode") or "full")
    client_key = f"local_mcp|{mcp_tool}|b_alias={backend_b_alias_mode}"
    client = _MCP_BY_KEY.get(client_key)
    if client is None:
        client = LocalMCPClient()
        client.register_search_logs(
            backend="backend_a",
            handler=search_logs_backend_a,
            aliases={
                "timestamp": "timestamp",
                "episode_id": "episode_id",
                "event_type": "event_type",
                "host": "host",
                "user": "user",
                "src_ip": "src_ip",
                "dst_ip": "dst_ip",
                "action": "action",
                "outcome": "outcome",
                "severity": "severity",
                "process_name": "process_name",
                "tags": "tags",
            },
            version="1.0",
        )
        client.register_search_logs(
            backend="backend_b",
            handler=search_logs_backend_b,
            aliases=(
                BACKEND_B_ALIASES_MINIMAL
                if backend_b_alias_mode == "minimal"
                else BACKEND_B_ALIASES_FULL
            ),
            version="1.0",
        )
        _MCP_BY_KEY[client_key] = client
    return client


def _search_logs(state: "BlueState", **kwargs: Any) -> Dict[str, Any]:
    backend = str(state.get("logs_backend") or "backend_a")
    logs_dir = str(state["logs_dir"])
    mcp_enabled = bool(state.get("mcp_enabled", True))
    mcp_tool = str(state.get("mcp_tool") or "search_logs")

    if mcp_enabled:
        try:
            client = get_mcp_client(state)
            return client.call_tool(
                tool_name=mcp_tool,
                backend=backend,
                logs_dir=logs_dir,
                **kwargs,
            )
        except Exception as exc:
            # Keep runs alive even if MCP dispatch fails.
            fallback_result = (
                search_logs_backend_b(logs_dir, **kwargs)
                if backend == "backend_b"
                else search_logs_backend_a(logs_dir, **kwargs)
            )
            out = dict(fallback_result)
            out["_tool_meta"] = {
                "mode": "legacy_fallback",
                "tool_name": mcp_tool,
                "backend": backend,
                "error": str(exc),
            }
            return out

    legacy_result = (
        search_logs_backend_b(logs_dir, **kwargs)
        if backend == "backend_b"
        else search_logs_backend_a(logs_dir, **kwargs)
    )
    out = dict(legacy_result)
    out["_tool_meta"] = {
        "mode": "legacy_direct",
        "tool_name": mcp_tool,
        "backend": backend,
    }
    return out


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


def _timing_enter(state: "BlueState", stage: str) -> Dict[str, Any]:
    timing = dict(state.get("timing") or {})
    stages = dict(timing.get("stages") or {})
    data = dict(stages.get(stage) or {})
    data["started_at"] = _iso_now()
    data["_t0_perf"] = time.perf_counter()
    stages[stage] = data
    timing["stages"] = stages
    if "_pipeline_t0_perf" not in timing:
        timing["_pipeline_t0_perf"] = time.perf_counter()
        timing["pipeline_started_at"] = _iso_now()
    return timing


def _timing_exit(timing: Dict[str, Any], stage: str) -> Dict[str, Any]:
    stages = dict(timing.get("stages") or {})
    data = dict(stages.get(stage) or {})
    t0 = data.pop("_t0_perf", None)
    data["finished_at"] = _iso_now()
    data["duration_ms"] = round((time.perf_counter() - float(t0)) * 1000.0, 3) if t0 is not None else None
    stages[stage] = data
    timing["stages"] = stages
    return timing


def _timing_finalize(timing: Dict[str, Any]) -> Dict[str, Any]:
    t0 = timing.pop("_pipeline_t0_perf", None)
    timing["pipeline_finished_at"] = _iso_now()
    timing["pipeline_duration_ms"] = round((time.perf_counter() - float(t0)) * 1000.0, 3) if t0 is not None else None
    return timing


def parse_iso_z(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_event(ev: Dict[str, Any], schema_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    mapping = schema_mapping or {}

    def read_source(source_spec: Optional[str]) -> Any:
        if not source_spec:
            return None
        candidates = [part.strip() for part in str(source_spec).split("|") if part.strip()]
        for source in candidates:
            if "." in source:
                cur: Any = ev
                ok = True
                for piece in source.split("."):
                    if not isinstance(cur, dict) or piece not in cur:
                        ok = False
                        break
                    cur = cur.get(piece)
                if ok and cur is not None:
                    return cur
                continue
            if ev.get(source) is not None:
                return ev.get(source)
        return None

    def pick(canonical: str) -> Any:
        mapped = mapping.get(canonical)
        picked_mapped = read_source(mapped)
        if picked_mapped is not None:
            return picked_mapped
        for alias in FALLBACK_ALIASES.get(canonical, [canonical]):
            picked_alias = read_source(alias)
            if picked_alias is not None:
                return picked_alias
        return None

    def normalize_timestamp(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        s = str(value).strip()
        if not s:
            return None
        if s.isdigit():
            ts = float(s)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            return parse_iso_z(s).isoformat().replace("+00:00", "Z")
        except ValueError:
            return s

    def normalize_event_type(value: Any) -> Any:
        if value is None:
            return None
        s = str(value).strip().lower()
        return {
            "authentication": "auth",
            "netflow": "network",
            "proc_event": "process",
        }.get(s, s)

    def normalize_action(value: Any) -> Any:
        if value is None:
            return None
        s = str(value).strip().lower()
        return {
            "auth_try": "login_attempt",
            "auth_ok": "login_success",
            "remote_auth_ok": "remote_auth_success",
            "svc_connect": "connect_remote_service",
            "proc_spawn": "process_start",
            "proc_exit": "process_end",
            "dns_lookup": "dns_query",
            "net_connect": "connect",
        }.get(s, s)

    def normalize_outcome(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return "success" if value else "fail"
        if isinstance(value, (int, float)):
            return "success" if float(value) > 0 else "fail"
        s = str(value).strip().lower()
        if s in {"ok", "success", "true", "1"}:
            return "success"
        if s in {"fail", "false", "0", "error"}:
            return "fail"
        return s

    def normalize_severity(value: Any) -> str:
        if value is None:
            return "low"
        if isinstance(value, (int, float)):
            n = int(value)
            if n >= 3:
                return "high"
            if n == 2:
                return "medium"
            return "low"
        s = str(value).strip().lower()
        if s in {"p1", "critical", "high"}:
            return "high"
        if s in {"p2", "medium", "med"}:
            return "medium"
        if s in {"p3", "low"}:
            return "low"
        return s or "low"

    tags = pick("tags")
    if tags is None:
        tags = []
    if isinstance(tags, str):
        if "," in tags:
            tags = [part.strip().lower() for part in tags.split(",") if part.strip()]
        elif "|" in tags:
            tags = [part.strip().lower() for part in tags.split("|") if part.strip()]
        else:
            tags = [tags.strip().lower()] if tags.strip() else []
    elif isinstance(tags, dict):
        vals = tags.get("values")
        if isinstance(vals, list):
            tags = [str(v).strip().lower() for v in vals if str(v).strip()]
        else:
            tags = [str(tags).lower()]
    elif not isinstance(tags, list):
        tags = [str(tags).strip().lower()]
    else:
        tags = [str(v).strip().lower() for v in tags if str(v).strip()]

    return {
        "timestamp": normalize_timestamp(pick("timestamp")),
        "episode_id": pick("episode_id"),
        "seed": pick("seed"),
        "event_type": normalize_event_type(pick("event_type")),
        "host": pick("host"),
        "user": pick("user"),
        "src_ip": pick("src_ip"),
        "dst_ip": pick("dst_ip"),
        "action": normalize_action(pick("action")),
        "outcome": normalize_outcome(pick("outcome")),
        "severity": normalize_severity(pick("severity")),
        "process_name": pick("process_name"),
        "tags": tags,
    }


def _event_timestamp(ev: Dict[str, Any]) -> str:
    raw_value = (
        ev.get("timestamp")
        or ev.get("time")
        or ev.get("when_utc")
        or ev.get("ts_utc")
        or ev.get("ts_epoch_ms")
        or ev.get("tstamp")
        or ev.get("time_obs")
    )
    if raw_value is None:
        return ""
    if isinstance(raw_value, (int, float)):
        ts = float(raw_value)
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    s = str(raw_value).strip()
    if not s:
        return ""
    if s.isdigit():
        ts = float(s)
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        return parse_iso_z(s).isoformat().replace("+00:00", "Z")
    except ValueError:
        return s


def _contract_alias_mapping_from_search_tool(state: "BlueState") -> Dict[str, str]:
    info = state.get("search_tool_info") or {}
    calls = info.get("calls") or []
    if not isinstance(calls, list):
        return {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        aliases = call.get("aliases")
        if isinstance(aliases, dict) and aliases:
            # aliases already come canonical -> backend_field
            return {str(k): str(v) for k, v in aliases.items()}
    return {}


def _event_hour_bucket(timestamp: Any) -> str:
    if not timestamp:
        return "unknown_hour"
    try:
        hour = parse_iso_z(str(timestamp)).hour
    except (TypeError, ValueError):
        return "unknown_hour"
    if 8 <= hour <= 18:
        return "business_hours"
    return "off_hours"


def _compact_token(value: Any, *, default: str = "none") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else default


def build_case_text(ev: Dict[str, Any], asset_ctx: Dict[str, Any]) -> str:
    asset = asset_ctx.get("asset") or {}
    tags = " ".join(ev.get("tags") or [])
    pattern_text = build_pattern_text(ev, asset_ctx)
    return (
        f"event_type={ev.get('event_type')} action={ev.get('action')} outcome={ev.get('outcome')} "
        f"severity={ev.get('severity')} user={ev.get('user')} src_ip={ev.get('src_ip')} "
        f"host={ev.get('host')} role={asset.get('role')} criticality={asset.get('criticality')} tags={tags} "
        f"{pattern_text}"
    )


def build_pattern_text(ev: Dict[str, Any], asset_ctx: Dict[str, Any]) -> str:
    asset = asset_ctx.get("asset") or {}
    tag_list = [str(tag).strip().lower() for tag in (ev.get("tags") or []) if str(tag).strip()]
    tags = " ".join(sorted(tag_list)) or "none"
    return (
        "pattern "
        f"event_type={_compact_token(ev.get('event_type'))} "
        f"action={_compact_token(ev.get('action'))} "
        f"outcome={_compact_token(ev.get('outcome'))} "
        f"user={_compact_token(ev.get('user'))} "
        f"host={_compact_token(ev.get('host'))} "
        f"role={_compact_token(asset.get('role'))} "
        f"criticality={_compact_token(asset.get('criticality'))} "
        f"hour_bucket={_event_hour_bucket(ev.get('timestamp'))} "
        f"tags={tags}"
    )


def build_pattern_key(ev: Dict[str, Any], asset_ctx: Dict[str, Any]) -> str:
    asset = asset_ctx.get("asset") or {}
    return "|".join(
        [
            _compact_token(ev.get("event_type")),
            _compact_token(ev.get("action")),
            _compact_token(ev.get("outcome")),
            _compact_token(ev.get("user")),
            _compact_token(ev.get("host")),
            _compact_token(asset.get("role")),
            _event_hour_bucket(ev.get("timestamp")),
        ]
    )


def _memory_summary(hits: List[Dict[str, Any]], *, current_pattern_key: Optional[str] = None) -> Dict[str, Any]:
    summary = {
        "count": len(hits),
        "tp_weight": 0.0,
        "fp_weight": 0.0,
        "uncertain_weight": 0.0,
        "top_label": None,
        "top_score": 0.0,
        "consensus": "none",
        "fp_recurrent_matches": 0,
        "fp_recurrent_weight": 0.0,
    }
    if not hits:
        return summary

    top = hits[0]
    summary["top_label"] = (top.get("case") or {}).get("label")
    summary["top_score"] = float(top.get("score", 0.0))

    for hit in hits:
        label = ((hit.get("case") or {}).get("label") or "").upper()
        score = float(hit.get("score", 0.0))
        raw_conf = ((hit.get("case") or {}).get("confidence"))
        try:
            case_conf = float(raw_conf) if raw_conf is not None else 0.75
        except (TypeError, ValueError):
            case_conf = 0.75
        case_conf = max(0.25, min(1.0, case_conf))
        weighted_score = score * case_conf
        tags = [str(tag).strip().lower() for tag in ((hit.get("case") or {}).get("tags") or [])]
        source_meta = (hit.get("case") or {}).get("source") or {}
        stored_pattern_key = str(source_meta.get("pattern_key") or "")
        exact_recurrent_fp = (
            label == "FP"
            and "recurrent_pattern" in tags
            and bool(current_pattern_key)
            and stored_pattern_key == current_pattern_key
        )
        if exact_recurrent_fp:
            weighted_score *= 1.35
            summary["fp_recurrent_matches"] += 1
            summary["fp_recurrent_weight"] += weighted_score
        if label == "TP":
            summary["tp_weight"] += weighted_score
        elif label == "FP":
            summary["fp_weight"] += weighted_score
        elif label == "UNCERTAIN":
            summary["uncertain_weight"] += weighted_score

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


def _merge_memory_hits(*hit_sets: List[Any], k: int = 3) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for hit_set in hit_sets:
        for hit in hit_set:
            case = hit.case
            case_id = int(case.get("case_id") or 0)
            if case_id <= 0:
                continue
            score = float(hit.score)
            current = merged.get(case_id)
            if current is None or score > float(current.get("score", 0.0)):
                merged[case_id] = {"score": score, "case": case}
    ordered = sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ordered[:k]


def _baseline_decision(severity: str, criticality: str, signals: int) -> Dict[str, Any]:
    if severity == "high" and criticality in ("high", "medium"):
        confidence = 0.90 if signals >= 2 else 0.80
        return {
            "decision": "block_ip",
            "confidence": confidence,
            "decision_reason": "Severidad high en activo critico: contener.",
        }

    if severity == "high" and signals >= 2:
        return {
            "decision": "block_ip",
            "confidence": 0.82,
            "decision_reason": "Severidad high con correlacion temporal suficiente: contener.",
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
    logs_backend: str
    episode_id: int
    response_delay_sec: int
    interactive: bool
    proposed_decision: str
    final_decision: str
    gating: Dict[str, Any]
    case_text: Optional[str]
    pattern_text: Optional[str]
    pattern_key: Optional[str]
    raw_events: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    detection_event: Optional[Dict[str, Any]]
    t_detect: Optional[str]
    t_detect_source: Optional[str]
    asset_context: Dict[str, Any]
    memory_hits: List[Dict[str, Any]]
    correlation: Dict[str, Any]
    run_id: Optional[str]
    decisions_path: Optional[str]
    actions_path: Optional[str]
    memory_dir: Optional[str]
    gt_dir: Optional[str]
    schema_mapper_mode: Optional[str]
    schema_map_min_confidence: Optional[float]
    schema_map_cache_path: Optional[str]
    schema_cache_scope: Optional[str]
    schema_adapt_mode: Optional[str]
    backend_b_alias_mode: Optional[str]
    mcp_enabled: Optional[bool]
    mcp_tool: Optional[str]
    llm_provider: Optional[str]
    gemini_api_key: Optional[str]
    gemini_model: Optional[str]
    ollama_url: Optional[str]
    ollama_model: Optional[str]
    llm_timeout_sec: Optional[float]
    schema_mapping: Dict[str, Any]
    search_tool_info: Dict[str, Any]
    decision: str
    confidence: float
    decision_reason: str
    decision_trace: Dict[str, Any]
    approved: bool
    action_result: Optional[Dict[str, Any]]
    timing: Dict[str, Any]


SUSPICIOUS_TAGS = ["suspicious", "lateral_like", "success_after_fail", "post_auth"]


def observe(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "observe")
    episode_id = state["episode_id"]

    result = _search_logs(state, episode_id=episode_id, filters={"tags_any": SUSPICIOUS_TAGS}, limit=200)
    first_tool_meta = result.pop("_tool_meta", None)
    events = result["events"]
    fallback_used = False
    calls: List[Dict[str, Any]] = []
    if isinstance(first_tool_meta, dict):
        calls.append({"stage": "observe_primary", **first_tool_meta})

    if not events:
        fallback = _search_logs(state, episode_id=episode_id, filters={"severity": "high"}, limit=200)
        fallback_tool_meta = fallback.pop("_tool_meta", None)
        events = fallback["events"]
        fallback_used = True
        if isinstance(fallback_tool_meta, dict):
            calls.append({"stage": "observe_fallback", **fallback_tool_meta})

    events.sort(key=_event_timestamp)

    detection_event = events[0] if events else None

    # Anchor MTTD on earliest plausible indicator tied to the same source IP.
    # This keeps the metric comparable while reducing bias from late confirmation events.
    indicator_tags = SUSPICIOUS_TAGS + ["burst", "auth"]
    early: List[Dict[str, Any]] = []
    src_ip = (
        (detection_event or {}).get("src_ip")
        or (detection_event or {}).get("origin_addr")
        or (detection_event or {}).get("source_ip")
    )
    if detection_event and src_ip:
        early_result = _search_logs(
            state,
            episode_id=episode_id,
            filters={"tags_any": indicator_tags, "src_ip": src_ip},
            limit=400,
        )
        early_tool_meta = early_result.pop("_tool_meta", None)
        if isinstance(early_tool_meta, dict):
            calls.append({"stage": "observe_early_anchor", **early_tool_meta})
        early = early_result["events"]
        early.sort(key=_event_timestamp)
    elif events and not fallback_used:
        early = list(events)
    else:
        early = _search_logs(
            state,
            episode_id=episode_id,
            filters={"tags_any": indicator_tags},
            limit=200,
        )["events"]
        early.sort(key=_event_timestamp)

    if early:
        t_detect = _event_timestamp(early[0]) or None
        t_detect_source = "earliest_indicator"
    elif detection_event:
        t_detect = _event_timestamp(detection_event) or None
        t_detect_source = "decision_event"
    else:
        t_detect = None
        t_detect_source = None

    if calls:
        stages = dict(timing.get("stages") or {})
        stage_data = dict(stages.get("observe") or {})
        stage_data["tool_mode"] = calls[0].get("mode")
        stage_data["tool_backend"] = calls[0].get("backend")
        stage_data["tool_name"] = calls[0].get("tool_name")
        stage_data["tool_calls"] = len(calls)
        stages["observe"] = stage_data
        timing["stages"] = stages

    timing = _timing_exit(timing, "observe")
    return {
        "raw_events": events,
        "detection_event": detection_event,
        "t_detect": t_detect,
        "t_detect_source": t_detect_source,
        "search_tool_info": {
            "tool_name": str(state.get("mcp_tool") or "search_logs"),
            "backend": str(state.get("logs_backend") or "backend_a"),
            "mcp_enabled": bool(state.get("mcp_enabled", True)),
            "fallback_used": fallback_used,
            "calls": calls,
        },
        "timing": timing,
    }


def normalize_schema(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "normalize_schema")
    raw = state.get("raw_events") or []
    mapper = get_schema_mapper(state)
    backend = str(state.get("logs_backend") or "backend_a")
    adapt_mode = str(state.get("schema_adapt_mode") or "contract_first")

    contract_mapping = _contract_alias_mapping_from_search_tool(state)
    contract_core_ok = all(
        field in contract_mapping for field in ("timestamp", "src_ip", "severity", "tags", "event_type")
    )

    mapping_result = None
    if raw and (adapt_mode == "llm_first" or not contract_core_ok):
        mapping_result = mapper.infer_mapping(
            backend=backend,
            sample_events=raw[:8],
            contract_hints=contract_mapping,
        )

    if adapt_mode == "contract_first" and contract_core_ok:
        # Fast-path: MCP contract already provides canonical aliases.
        mapping = dict(contract_mapping)
        mapping_source = "contract_alias_direct"
        mapping_confidence = 1.0
        mapping_signature = ""
        mapping_cache_hit = True
        mapping_llm_called = False
        mapping_error = ""
    else:
        mapping = dict(mapping_result.mapping if mapping_result is not None else {})
        mapping_source = mapping_result.source if mapping_result is not None else "none"
        mapping_confidence = float(mapping_result.confidence if mapping_result is not None else 0.0)
        mapping_signature = mapping_result.signature if mapping_result is not None else ""
        mapping_cache_hit = bool(mapping_result.cache_hit) if mapping_result is not None else False
        mapping_llm_called = bool(mapping_result.llm_called) if mapping_result is not None else False
        mapping_error = str(mapping_result.error or "") if mapping_result is not None else ""

    missing_core = any(
        field not in mapping for field in ("timestamp", "src_ip", "severity", "tags", "event_type")
    )
    if (
        contract_mapping
        and missing_core
        and (str(mapping_source).startswith("fallback") or mapping_source in {"none", "empty"})
    ):
        # Use the MCP contract aliases as a safety net when dynamic mapping fails.
        merged = dict(contract_mapping)
        merged.update(mapping)
        mapping = merged
        mapping_source = "contract_alias_fallback"
        mapping_confidence = max(mapping_confidence, 0.65)

    events = [normalize_event(event, mapping) for event in raw]
    detection_event = state.get("detection_event")
    detection_event_norm = normalize_event(detection_event, mapping) if detection_event else None
    t_detect = state.get("t_detect")
    t_detect_source = state.get("t_detect_source")
    if not t_detect and detection_event_norm and detection_event_norm.get("timestamp"):
        t_detect = detection_event_norm.get("timestamp")
        if not t_detect_source:
            t_detect_source = "normalized_detection_event"

    stages = dict(timing.get("stages") or {})
    stage_data = dict(stages.get("normalize_schema") or {})
    stage_data["backend"] = backend
    stage_data["mapper_enabled"] = bool(mapper.enabled)
    stage_data["llm_provider"] = str(mapper.provider)
    stage_data["mapping_source"] = mapping_source
    stage_data["mapping_confidence"] = mapping_confidence
    stage_data["mapping_signature"] = mapping_signature
    stage_data["cache_hit"] = mapping_cache_hit
    stage_data["llm_called"] = mapping_llm_called
    if mapping_error:
        stage_data["mapping_error"] = mapping_error
    stage_data["adapt_mode"] = adapt_mode
    stage_data["backend_b_alias_mode"] = str(state.get("backend_b_alias_mode") or "full")
    stages["normalize_schema"] = stage_data
    timing["stages"] = stages

    timing = _timing_exit(timing, "normalize_schema")
    return {
        "events": events,
        "detection_event": detection_event_norm,
        "t_detect": t_detect,
        "t_detect_source": t_detect_source,
        "schema_mapping": {
            "backend": backend,
            "mapper_enabled": bool(mapper.enabled),
            "llm_provider": str(mapper.provider),
            "source": mapping_source,
            "confidence": mapping_confidence,
            "signature": mapping_signature,
            "cache_hit": mapping_cache_hit,
            "llm_called": mapping_llm_called,
            "error": mapping_error or None,
            "adapt_mode": adapt_mode,
            "backend_b_alias_mode": str(state.get("backend_b_alias_mode") or "full"),
            "mapping": mapping,
        },
        "timing": timing,
    }


def enrich(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "enrich")
    detection_event = state.get("detection_event")
    if not detection_event:
        timing = _timing_exit(timing, "enrich")
        return {"asset_context": {"found": False, "notes": ["no_detection_event"]}, "timing": timing}
    host = detection_event.get("host") or ""
    timing = _timing_exit(timing, "enrich")
    return {"asset_context": get_asset_context(host), "timing": timing}


def retrieve_memory(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "retrieve_memory")
    detection_event = state.get("detection_event")
    asset_context = state.get("asset_context") or {}
    if not detection_event:
        timing = _timing_exit(timing, "retrieve_memory")
        return {"memory_hits": [], "case_text": None, "pattern_text": None, "timing": timing}

    case_text = build_case_text(detection_event, asset_context)
    pattern_text = build_pattern_text(detection_event, asset_context)
    pattern_key = build_pattern_key(detection_event, asset_context)
    mem = get_memory(_memory_dir_from_state(state))
    case_hits = mem.search(text=case_text, k=4, threshold=MEMORY_SEARCH_THRESHOLD)
    pattern_hits = mem.search(text=pattern_text, k=4, threshold=max(0.56, MEMORY_SEARCH_THRESHOLD - 0.06))
    hits = _merge_memory_hits(case_hits, pattern_hits, k=3)
    timing = _timing_exit(timing, "retrieve_memory")
    return {
        "memory_hits": hits,
        "case_text": case_text,
        "pattern_text": pattern_text,
        "pattern_key": pattern_key,
        "timing": timing,
    }


def correlate(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "correlate")
    detection_event = state.get("detection_event")
    search_tool_info = dict(state.get("search_tool_info") or {})
    calls = list(search_tool_info.get("calls") or [])
    if not detection_event:
        timing = _timing_exit(timing, "correlate")
        return {"correlation": {"signals": 0, "summary": "No detection event."}, "timing": timing}

    episode_id = state["episode_id"]
    src_ip = detection_event.get("src_ip")
    t_detect = detection_event.get("timestamp")
    signals = 1
    window: Dict[str, Any] = {}

    if src_ip and t_detect:
        t0 = parse_iso_z(t_detect)
        start = iso_z(t0 - timedelta(minutes=2))
        end = iso_z(t0 + timedelta(minutes=2))
        result = _search_logs(
            state,
            episode_id=episode_id,
            start=start,
            end=end,
            filters={"src_ip": src_ip},
            limit=200,
            agg={"type": "count"},
        )
        tool_meta = result.pop("_tool_meta", None)
        if isinstance(tool_meta, dict):
            calls.append({"stage": "correlate", **tool_meta})
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
    if calls:
        search_tool_info["calls"] = calls
        search_tool_info["total_calls"] = len(calls)
    timing = _timing_exit(timing, "correlate")
    return {"correlation": correlation, "search_tool_info": search_tool_info, "timing": timing}


def decide(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "decide")
    detection_event = state.get("detection_event")
    if not detection_event:
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "no_block",
            "confidence": 0.20,
            "decision_reason": "Sin evento de deteccion: no se actua.",
            "timing": timing,
        }

    asset = (state.get("asset_context") or {}).get("asset") or {}
    criticality = asset.get("criticality", "low")
    tags = detection_event.get("tags") or []
    severity = detection_event.get("severity", "low")
    signals = (state.get("correlation") or {}).get("signals", 1)

    allowlisted = ("allowlisted_user" in tags) or ("service_account" in tags)
    if allowlisted:
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "no_block",
            "confidence": 0.95,
            "decision_reason": "Allowlisted/service_account: evitar falso positivo.",
            "decision_trace": {
                "base_decision": "no_block",
                "base_confidence": 0.95,
                "memory_consensus": "none",
                "memory_top_score": 0.0,
                "memory_tp_weight": 0.0,
                "memory_fp_weight": 0.0,
                "memory_rule_applied": "allowlist_short_circuit",
            },
            "timing": timing,
        }

    base = _baseline_decision(severity, criticality, signals)
    decision_trace = {
        "base_decision": str(base["decision"]),
        "base_confidence": float(base["confidence"]),
        "memory_consensus": "none",
        "memory_top_score": 0.0,
        "memory_tp_weight": 0.0,
        "memory_fp_weight": 0.0,
        "memory_rule_applied": "no_memory_hits",
    }
    hits = state.get("memory_hits") or []
    if not hits:
        timing = _timing_exit(timing, "decide")
        base["timing"] = timing
        base["decision_trace"] = decision_trace
        return base

    memory = _memory_summary(hits, current_pattern_key=state.get("pattern_key"))
    top_score = float(memory["top_score"])
    tp_weight = float(memory["tp_weight"])
    fp_weight = float(memory["fp_weight"])
    base_decision = str(base["decision"])
    base_confidence = float(base["confidence"])
    event_type = str((detection_event or {}).get("event_type") or "")
    decision_trace = {
        "base_decision": base_decision,
        "base_confidence": base_confidence,
        "memory_consensus": str(memory["consensus"]),
        "memory_top_score": top_score,
        "memory_tp_weight": tp_weight,
        "memory_fp_weight": fp_weight,
        "memory_fp_recurrent_matches": int(memory["fp_recurrent_matches"]),
        "memory_rule_applied": "memory_observed_no_override",
    }

    # Memory-gated policy for high-severity network events on low-criticality assets:
    # use memory as confirmation before hard containment to make FAISS influence explicit.
    if (
        base_decision == "block_ip"
        and severity == "high"
        and criticality == "low"
        and signals >= 2
        and event_type == "network"
    ):
        if (
            memory["consensus"] == "TP"
            and top_score >= 0.68
            and tp_weight >= (fp_weight + 0.01)
        ):
            timing = _timing_exit(timing, "decide")
            return {
                "decision": "block_ip",
                "confidence": max(base_confidence, 0.79),
                "decision_reason": (
                    f"Evento de red high en activo low-criticality: memoria confirma TP (score={top_score:.2f}) "
                    "y habilita contencion."
                ),
                "decision_trace": {
                    **decision_trace,
                    "base_decision": "escalate",
                    "base_confidence": 0.68,
                    "memory_rule_applied": "memory_gate_promote_block",
                },
                "timing": timing,
            }
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "block_ip",
            "confidence": max(base_confidence, 0.80),
            "decision_reason": (
                f"{base['decision_reason']} | Memoria no supera umbral de confirmacion para gate de red; "
                "se conserva contencion por evidencia actual."
            ),
            "decision_trace": {
                **decision_trace,
                "memory_rule_applied": "memory_gate_no_promote_keep_block",
            },
            "timing": timing,
        }

    if memory["consensus"] == "TP" and top_score >= MEMORY_TP_STRONG_THRESHOLD and tp_weight >= (fp_weight + MEMORY_TP_MARGIN):
        if base_decision == "escalate":
            timing = _timing_exit(timing, "decide")
            return {
                "decision": "block_ip",
                "confidence": max(base_confidence, 0.85),
                "decision_reason": f"Memoria respalda TP fuerte (score={top_score:.2f}): contener.",
                "decision_trace": {**decision_trace, "memory_rule_applied": "tp_strong_promote_to_block"},
                "timing": timing,
            }
        timing = _timing_exit(timing, "decide")
        return {
            "decision": base_decision,
            "confidence": max(base_confidence, 0.88),
            "decision_reason": f"{base['decision_reason']} | Memoria consistente con TP (score={top_score:.2f}).",
            "decision_trace": {**decision_trace, "memory_rule_applied": "tp_strong_reinforce"},
            "timing": timing,
        }

    if (
        memory["consensus"] == "TP"
        and base_decision == "escalate"
        and top_score >= MEMORY_TP_PROMOTION_THRESHOLD
        and tp_weight >= fp_weight
        and (signals >= 2 or severity == "high")
    ):
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "block_ip",
            "confidence": max(base_confidence, 0.80),
            "decision_reason": f"Evidencia actual suficiente y memoria con TP similar (score={top_score:.2f}): promover a contencion.",
            "decision_trace": {**decision_trace, "memory_rule_applied": "tp_medium_promote_with_signals"},
            "timing": timing,
        }

    if (
        memory["consensus"] == "TP"
        and base_decision == "escalate"
        and top_score >= 0.68
        and tp_weight >= (fp_weight + 0.01)
        and (severity in {"high", "medium"})
        and signals >= 1
    ):
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "block_ip",
            "confidence": max(base_confidence, 0.78),
            "decision_reason": f"Memoria sugiere TP util (score={top_score:.2f}) y evidencia activa: promocion suave a contencion.",
            "decision_trace": {**decision_trace, "memory_rule_applied": "tp_soft_promote_to_block"},
            "timing": timing,
        }

    fp_recurrent_match_count = int(memory["fp_recurrent_matches"])
    fp_recurrent_match = fp_recurrent_match_count > 0
    fp_threshold = 0.78 if fp_recurrent_match else MEMORY_FP_STRONG_THRESHOLD
    if memory["consensus"] == "FP" and top_score >= fp_threshold and fp_weight >= (tp_weight + MEMORY_FP_MARGIN):
        if base_decision == "block_ip":
            recurrent_safe_suppress = (
                fp_recurrent_match
                and fp_recurrent_match_count >= 1
                and top_score >= 0.94
                and criticality == "low"
                and signals <= 2
                and str((detection_event or {}).get("outcome") or "") == "success"
            )
            if recurrent_safe_suppress:
                timing = _timing_exit(timing, "decide")
                return {
                    "decision": "no_block",
                    "confidence": 0.90,
                    "decision_reason": (
                        f"Memoria reconoce patron benigno recurrente exacto (score={top_score:.2f}) "
                        "en activo low-criticality: suprimir accion."
                    ),
                    "decision_trace": {
                        **decision_trace,
                        "memory_rule_applied": "fp_recurrent_exact_suppress_action",
                    },
                    "timing": timing,
                }
            if severity == "high" and criticality in ("high", "medium") and signals >= 2:
                timing = _timing_exit(timing, "decide")
                return {
                    "decision": "block_ip",
                    "confidence": min(base_confidence, 0.82),
                    "decision_reason": f"{base['decision_reason']} | Memoria sugiere FP, pero la evidencia actual sigue siendo fuerte.",
                    "decision_trace": {**decision_trace, "memory_rule_applied": "fp_strong_but_current_evidence_kept_block"},
                    "timing": timing,
                }
            timing = _timing_exit(timing, "decide")
            return {
                "decision": "escalate",
                "confidence": 0.76 if fp_recurrent_match else 0.72,
                "decision_reason": (
                    f"Memoria sugiere FP similar (score={top_score:.2f}): pedir mayor evidencia antes de contener."
                    if not fp_recurrent_match
                    else f"Memoria reconoce patron benigno recurrente (score={top_score:.2f}): degradar a escalamiento."
                ),
                "decision_trace": {
                    **decision_trace,
                    "memory_rule_applied": "fp_recurrent_downgrade_to_escalate" if fp_recurrent_match else "fp_strong_downgrade_to_escalate",
                },
                "timing": timing,
            }
        timing = _timing_exit(timing, "decide")
        return {
            "decision": "no_block",
            "confidence": 0.88 if fp_recurrent_match else 0.82,
            "decision_reason": (
                f"Memoria sugiere FP similar (score={top_score:.2f}) y la evidencia actual es debil."
                if not fp_recurrent_match
                else f"Memoria reconoce patron benigno recurrente (score={top_score:.2f}) y suprime accion."
            ),
            "decision_trace": {
                **decision_trace,
                "memory_rule_applied": "fp_recurrent_suppress_action" if fp_recurrent_match else "fp_strong_suppress_action",
            },
            "timing": timing,
        }

    if memory["count"] >= 2:
        timing = _timing_exit(timing, "decide")
        return {
            "decision": base_decision,
            "confidence": max(0.55, base_confidence - 0.08),
            "decision_reason": f"{base['decision_reason']} | Memoria mixta o insuficiente; se prioriza la heuristica base.",
            "decision_trace": {**decision_trace, "memory_rule_applied": "memory_mixed_keep_base"},
            "timing": timing,
        }

    timing = _timing_exit(timing, "decide")
    base["timing"] = timing
    base["decision_trace"] = decision_trace
    return base


def act(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "act")
    proposed = state.get("decision", "no_block")
    detection_event = state.get("detection_event") or {}
    t_detect = state.get("t_detect") or detection_event.get("timestamp")
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

    timing = _timing_exit(timing, "act")
    return {
        "proposed_decision": proposed,
        "final_decision": final_decision,
        "approved": approved,
        "gating": gating,
        "action_result": action_result,
        "timing": timing,
    }


def log(state: BlueState) -> BlueState:
    timing = _timing_enter(state, "log")
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
        detection_event = state.get("detection_event") or {}
        pattern_text = state.get("pattern_text") or build_pattern_text(detection_event, state.get("asset_context") or {})
        pattern_key = state.get("pattern_key") or build_pattern_key(detection_event, state.get("asset_context") or {})

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
                if detection_event.get("user") or detection_event.get("host") or detection_event.get("action"):
                    learned_tags.append("recurrent_pattern")

        if gating.get("prompted") is True and label is None:
            label = "TP" if approved else "FP"
            learned_decision = final
            learned_reason = f"gating_feedback: approved={approved}"
            learned_tags = ["gating_feedback"]

        if label and learned_decision and learned_reason:
            mem = get_memory(_memory_dir_from_state(state))
            stored_text = f"{case_text} {pattern_text}".strip()
            if not _memory_case_exists(mem, text=stored_text, label=label, episode_id=episode_id):
                mem.add_case(
                    text=stored_text,
                    label=label,
                    decision=learned_decision,
                    reason=learned_reason,
                    tags=learned_tags,
                    confidence=state.get("confidence"),
                    source={"episode_id": episode_id, "run_id": run_id, "pattern_key": pattern_key},
                )

    timing = _timing_exit(timing, "log")
    timing = _timing_finalize(timing)
    evidence = {
        "run_id": run_id,
        "proposed_decision": proposed,
        "final_decision": final,
        "t_detect_source": state.get("t_detect_source"),
        "gating": gating,
        "approved": approved,
        "detection_event": state.get("detection_event"),
        "asset_context": state.get("asset_context"),
        "memory_hits": state.get("memory_hits"),
        "decision_trace": state.get("decision_trace"),
        "schema_mapping": state.get("schema_mapping"),
        "search_tool": state.get("search_tool_info"),
        "correlation": state.get("correlation"),
        "action_result": state.get("action_result"),
        "timing": timing,
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

    return {"timing": timing}


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
