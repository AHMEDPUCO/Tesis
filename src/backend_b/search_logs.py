from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _parse_iso_z(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _episode_file(logs_dir: str, episode_id: int) -> str:
    return os.path.join(logs_dir, f"episode_{episode_id:03d}.jsonl")


def _pick(raw: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if "." in key:
            cur: Any = raw
            ok = True
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    ok = False
                    break
                cur = cur.get(part)
            if ok and cur is not None:
                return cur
            continue
        if raw.get(key) is not None:
            return raw.get(key)
    return None


def _to_iso_utc(value: Any) -> Optional[str]:
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
        return _parse_iso_z(s).isoformat().replace("+00:00", "Z")
    except ValueError:
        return s


def _norm_event_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return {
        "authentication": "auth",
        "netflow": "network",
        "proc_event": "process",
    }.get(s, s)


def _norm_action(value: Any) -> Optional[str]:
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


def _norm_outcome(value: Any) -> Optional[str]:
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


def _norm_severity(value: Any) -> str:
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


def _norm_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, dict):
        vals = value.get("values")
        if isinstance(vals, list):
            return [str(v).strip().lower() for v in vals if str(v).strip()]
        return [str(value).strip().lower()]
    s = str(value).strip()
    if not s:
        return []
    if "," in s:
        return [part.strip().lower() for part in s.split(",") if part.strip()]
    if "|" in s:
        return [part.strip().lower() for part in s.split("|") if part.strip()]
    return [s.lower()]


def _to_canonical(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": _to_iso_utc(_pick(raw, "when_utc", "ts_epoch_ms", "tstamp", "time_obs")),
        "episode_id": _pick(raw, "case_ref", "case_num", "incident_case", "ep_ref"),
        "seed": _pick(raw, "rnd", "rand_seed", "seed_id", "rnd_id"),
        "event_type": _norm_event_type(_pick(raw, "evt_kind", "evt_type_name", "kind", "cat")),
        "host": _pick(raw, "asset_ref", "asset_name", "node", "host_ref"),
        "user": _pick(raw, "actor_id", "principal", "account", "usr_ref"),
        "src_ip": _pick(raw, "origin_addr", "src_addr", "ip_from", "src"),
        "dst_ip": _pick(raw, "target_addr", "dst_addr", "ip_to", "dst"),
        "action": _norm_action(_pick(raw, "op_name", "op", "verb", "op_name_v2")),
        "outcome": _norm_outcome(_pick(raw, "result_state", "ok", "result_code", "state_text")),
        "severity": _norm_severity(_pick(raw, "risk_code", "sev_level", "priority", "risk")),
        "process_name": _pick(raw, "proc_image", "proc", "image", "proc_path", "proc_meta.image"),
        "tags": _norm_tags(_pick(raw, "labels_v2", "tag_blob", "labels", "tagset")),
    }


def _match_filters(ev: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    for key in [
        "host",
        "user",
        "src_ip",
        "dst_ip",
        "event_type",
        "action",
        "outcome",
        "severity",
        "process_name",
    ]:
        if key in filters and filters[key] is not None and ev.get(key) != filters[key]:
            return False

    tags = ev.get("tags") or []
    if "tags_any" in filters and filters["tags_any"]:
        if not any(tag in tags for tag in filters["tags_any"]):
            return False
    if "tags_all" in filters and filters["tags_all"]:
        if not all(tag in tags for tag in filters["tags_all"]):
            return False
    return True


def _match_query(ev: Dict[str, Any], query: Optional[str]) -> bool:
    if not query:
        return True

    q = query.strip()
    if ":" in q:
        field, value = q.split(":", 1)
        field = field.strip()
        value = value.strip()
        if field == "tags":
            return value in (ev.get("tags") or [])
        return str(ev.get(field, "")).strip() == value

    haystack = " ".join(
        [
            str(ev.get("event_type", "")),
            str(ev.get("host", "")),
            str(ev.get("user", "")),
            str(ev.get("src_ip", "")),
            str(ev.get("dst_ip", "")),
            str(ev.get("action", "")),
            str(ev.get("outcome", "")),
            str(ev.get("severity", "")),
            str(ev.get("process_name", "")),
            " ".join(ev.get("tags") or []),
        ]
    ).lower()
    return q.lower() in haystack


def _in_time_range(ev: Dict[str, Any], start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    ts = ev.get("timestamp")
    if not ts:
        return False
    t = _parse_iso_z(ts)
    if start and t < _parse_iso_z(start):
        return False
    if end and t > _parse_iso_z(end):
        return False
    return True


def search_logs(
    logs_dir: str,
    *,
    episode_id: Optional[int] = None,
    query: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    agg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    filters = filters or {}
    limit = max(0, int(limit))

    if episode_id is not None:
        path = _episode_file(logs_dir, int(episode_id))
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe archivo de episodio: {path}")
        paths = [path]
    else:
        paths = sorted(
            os.path.join(logs_dir, name)
            for name in os.listdir(logs_dir)
            if name.startswith("episode_") and name.endswith(".jsonl")
        )

    matched = 0
    events_out: List[Dict[str, Any]] = []
    counter = Counter()

    for path in paths:
        for raw in _iter_jsonl(path):
            ev = _to_canonical(raw)
            if not _in_time_range(ev, start, end):
                continue
            if not _match_filters(ev, filters):
                continue
            if not _match_query(ev, query):
                continue
            matched += 1

            if agg and agg.get("type") == "top_k":
                field = agg.get("field")
                if field:
                    counter[str(ev.get(field))] += 1

            if limit > 0 and len(events_out) < limit:
                events_out.append(raw)

    aggregation = None
    if agg:
        if agg.get("type") == "count":
            aggregation = {"count": matched}
        elif agg.get("type") == "top_k":
            k = int(agg.get("k", 10))
            aggregation = {"top_k": counter.most_common(k)}

    return {
        "matched": matched,
        "returned": len(events_out),
        "events": events_out,
        "aggregation": aggregation,
    }
