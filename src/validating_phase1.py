#!/usr/bin/env python3
"""
Validador Fase 1:
- Lee logs JSONL por episodio (Backend A)
- Resume severidades y tags
- Verifica mínimos: variedad de severidad y presencia de tags "útiles"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_episode(path: str) -> Dict:
    severity = Counter()
    event_type = Counter()
    actions = Counter()
    tags = Counter()
    users = Counter()
    hosts = Counter()

    total = 0
    for ev in iter_jsonl(path):
        total += 1
        severity[ev.get("severity", "missing")] += 1
        event_type[ev.get("event_type", "missing")] += 1
        actions[ev.get("action", "missing")] += 1
        users[ev.get("user", "none")] += 1
        hosts[ev.get("host", "missing")] += 1

        for t in (ev.get("tags") or []):
            tags[t] += 1

    return {
        "file": os.path.basename(path),
        "total": total,
        "severity": severity,
        "event_type": event_type,
        "actions": actions,
        "tags": tags,
        "top_users": users.most_common(5),
        "top_hosts": hosts.most_common(5),
    }


def pct(part: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(part/total)*100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="data/logs_backend_a", help="Carpeta con episode_XXX.jsonl")
    ap.add_argument("--min-medium", type=int, default=1, help="Mínimo de eventos medium por episodio (sanity)")
    ap.add_argument("--min-high", type=int, default=1, help="Mínimo de eventos high por episodio (sanity)")
    ap.add_argument("--print-top-tags", type=int, default=10, help="Cuántos tags top imprimir")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs_dir, "episode_*.jsonl")))
    if not paths:
        raise SystemExit(f"No se encontraron archivos en: {args.logs_dir}")

    global_sev = Counter()
    global_tags = Counter()
    global_types = Counter()

    # Tags "útiles" que esperas ver si aplicaste reglas
    useful_tags = {
        "auth_fail", "service_account", "allowlisted_user",
        "uncommon_process", "maintenance_window", "burst",
        "noisy_process", "network_noise", "missing_process_name"
    }

    print(f"\n=== VALIDACIÓN FASE 1 ===")
    print(f"Directorio: {args.logs_dir}")
    print(f"Episodios encontrados: {len(paths)}\n")

    failures: List[str] = []

    for p in paths:
        s = summarize_episode(p)

        total = s["total"]
        sev = s["severity"]
        tags = s["tags"]

        global_sev.update(sev)
        global_tags.update(tags)
        global_types.update(s["event_type"])

        # Sanity checks
        med = sev.get("medium", 0)
        high = sev.get("high", 0)

        # Nota: puede haber episodios con 0 high si tu inyección no marca high;
        # en ese caso baja el threshold o ajusta escenarios.
        if med < args.min_medium:
            failures.append(f"{s['file']}: medium={med} (<{args.min_medium})")
        if high < args.min_high:
            failures.append(f"{s['file']}: high={high} (<{args.min_high})")

        # Conteo de tags útiles presentes
        present_useful = sum(tags[t] for t in useful_tags if t in tags)

        print(f"- {s['file']} | total={total} | "
              f"low={sev.get('low',0)} ({pct(sev.get('low',0), total)}) | "
              f"medium={med} ({pct(med, total)}) | "
              f"high={high} ({pct(high, total)}) | "
              f"useful_tags={present_useful}")

    print("\n=== RESUMEN GLOBAL ===")
    total_all = sum(global_sev.values())
    print(f"Eventos totales: {total_all}")
    print(f"Severidad: low={global_sev.get('low',0)} ({pct(global_sev.get('low',0), total_all)}), "
          f"medium={global_sev.get('medium',0)} ({pct(global_sev.get('medium',0), total_all)}), "
          f"high={global_sev.get('high',0)} ({pct(global_sev.get('high',0), total_all)})")

    print("\nTop event_type:")
    for k, v in global_types.most_common(6):
        print(f"  - {k}: {v} ({pct(v, total_all)})")

    print(f"\nTop {args.print_top_tags} tags:")
    for k, v in global_tags.most_common(args.print_top_tags):
        print(f"  - {k}: {v} ({pct(v, total_all)})")

    present_useful_global = sorted([t for t in useful_tags if t in global_tags])
    print(f"\nTags útiles presentes ({len(present_useful_global)}): {present_useful_global}")

    if failures:
        print("\n=== ALERTAS (sanity checks) ===")
        for f in failures[:30]:
            print("  !", f)
        if len(failures) > 30:
            print(f"  ... {len(failures)-30} más")
        print("\nSugerencia: si esto falla, baja --min-high/--min-medium o ajusta la inyección/scoring.")
    else:
        print("\n✅ OK: La Fase 1 parece consistente (hay variación y distribución razonable).")


if __name__ == "__main__":
    main()
