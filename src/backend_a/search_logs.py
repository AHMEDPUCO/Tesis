from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter

# -----------------------
# Helpers
# -----------------------

def _parse_iso_z(ts: str) -> datetime:
    # "2026-02-19T10:28:31Z" -> datetime UTC
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _episode_file(logs_dir: str, episode_id: int) -> str:
    return os.path.join(logs_dir, f"episode_{episode_id:03d}.jsonl")

def _match_filters(ev: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    filters soportados (exact match salvo tags):
      host, user, src_ip, dst_ip, event_type, action, outcome, severity, process_name
      tags_any: ["tag1","tag2"]  -> match si evento contiene cualquiera
      tags_all: ["tag1","tag2"]  -> match si evento contiene todas
    """
    if not filters:
        return True

    for k in ["host","user","src_ip","dst_ip","event_type","action","outcome","severity","process_name"]:
        if k in filters and filters[k] is not None:
            if ev.get(k) != filters[k]:
                return False

    tags = ev.get("tags") or []
    if "tags_any" in filters and filters["tags_any"]:
        if not any(t in tags for t in filters["tags_any"]):
            return False
    if "tags_all" in filters and filters["tags_all"]:
        if not all(t in tags for t in filters["tags_all"]):
            return False

    return True

def _match_query(ev: Dict[str, Any], query: Optional[str]) -> bool:
    """
    Query simple:
    - None/"" => True
    - "term" => búsqueda substring en campos clave
    - "field:value" => match exacto para field o tags
    """
    if not query:
        return True

    q = query.strip()
    if ":" in q:
        field, value = q.split(":", 1)
        field, value = field.strip(), value.strip()
        if field == "tags":
            return value in (ev.get("tags") or [])
        return str(ev.get(field, "")).strip() == value

    haystack = " ".join([
        str(ev.get("event_type","")),
        str(ev.get("host","")),
        str(ev.get("user","")),
        str(ev.get("src_ip","")),
        str(ev.get("dst_ip","")),
        str(ev.get("action","")),
        str(ev.get("outcome","")),
        str(ev.get("severity","")),
        str(ev.get("process_name","")),
        " ".join(ev.get("tags") or []),
    ]).lower()

    return q.lower() in haystack

def _in_time_range(ev: Dict[str, Any], start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    t = _parse_iso_z(ev["timestamp"])
    if start:
        if t < _parse_iso_z(start):
            return False
    if end:
        if t > _parse_iso_z(end):
            return False
    return True

# -----------------------
# API principal (tool)
# -----------------------

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
    """
    Retorna:
      {
        "matched": int,
        "returned": int,
        "events": [ ... ],
        "aggregation": {...} | None
      }

    agg soportado:
      {"type":"count"}
      {"type":"top_k", "field":"src_ip", "k":10}
    """
    filters = filters or {}
    limit = max(0, int(limit))

    paths: List[str] = []
    if episode_id is not None:
        p = _episode_file(logs_dir, int(episode_id))
        if not os.path.exists(p):
            raise FileNotFoundError(f"No existe archivo de episodio: {p}")
        paths = [p]
    else:
        # Escanea todos los episodios (más lento)
        paths = sorted(
            os.path.join(logs_dir, fn) for fn in os.listdir(logs_dir)
            if fn.startswith("episode_") and fn.endswith(".jsonl")
        )

    matched = 0
    events_out: List[Dict[str, Any]] = []
    counter = Counter()

    for p in paths:
        for ev in _iter_jsonl(p):
            if not _in_time_range(ev, start, end):
                continue
            if not _match_filters(ev, filters):
                continue
            if not _match_query(ev, query):
                continue

            matched += 1

            # agregación
            if agg and agg.get("type") == "top_k":
                field = agg.get("field")
                if field:
                    counter[str(ev.get(field))] += 1

            # retorno limitado
            if limit == 0:
                continue
            if len(events_out) < limit:
                events_out.append(ev)

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

# -----------------------
# Demo rápido
# -----------------------

if __name__ == "__main__":
    # Ejemplo: busca auth_fail en episodio 2
    res = search_logs(
        "data/logs_backend_a",
        episode_id=2,
        filters={"event_type": "auth", "tags_any": ["auth_fail"]},
        limit=5,
        agg={"type": "top_k", "field": "user", "k": 5},
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


###python src/backend_a/search_logs.py COMANDO PARA CORRER EL ARCHIVO

