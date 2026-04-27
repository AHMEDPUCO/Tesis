#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.config import ASSETS, USERS
from core.models import Event, GroundTruth
from core.scenarios import SCENARIOS


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_asset(rng: random.Random, role: Optional[str] = None):
    cands = ASSETS if role is None else [a for a in ASSETS if a.role == role]
    return rng.choice(cands)


def get_asset_by_host(host: str):
    for asset in ASSETS:
        if asset.host == host:
            return asset
    raise KeyError(f"Unknown asset host: {host}")

# --- Reglas simples para variar severidad/tags en ruido benigno ---

SERVICE_ACCOUNTS = {"svc_backup"}
SCANNER_IPS = {"10.0.10.99"}          # IPs benignas que generan ruido (escáner interno)
MAINTENANCE_HOSTS = {"web-01"}        # hosts con mantenimiento frecuente
ALLOWLIST_USERS = {"svc_backup"}      # allowlist simple por usuario

COMMON_PROCESSES = {"chrome.exe", "systemd", "sshd", "python"}
SUSPICIOUS_BUT_BENIGN = {"python"}    # ejemplo: genera FP en algunos contextos (opcional)

RECURRENT_BENIGN_PROFILES: Tuple[Dict[str, str], ...] = (
    {
        "name": "benign_recurrent_inventory_sync",
        "user": "alice",
        "host": "ws-01",
        "peer_host": "db-01",
        "process_name": "python",
        "pattern_tag": "inventory_sync",
    },
    {
        "name": "benign_recurrent_support_bundle",
        "user": "bob",
        "host": "ws-02",
        "peer_host": "web-01",
        "process_name": "python",
        "pattern_tag": "support_bundle",
    },
    {
        "name": "benign_recurrent_admin_healthcheck",
        "user": "admin",
        "host": "ws-01",
        "peer_host": "jump-01",
        "process_name": "python",
        "pattern_tag": "admin_healthcheck",
    },
)

def assign_noise_severity_and_tags(event_type: str, action: str, outcome: str,
                                   user: str | None, process_name: str | None) -> tuple[str, list[str]]:
    tags = ["benign"]

    # Base
    severity = "low"

    # 1) Auth failures generan más severidad (ruido que parece ataque)
    if event_type == "auth" and action == "login_attempt" and outcome == "fail":
        severity = "medium"
        tags += ["auth_fail"]

    # 2) Service accounts suelen ser benignas pero ruidosas
    if user in SERVICE_ACCOUNTS:
        tags += ["service_account"]
        # si además falló auth, que quede medium (ya está)
        if severity == "low":
            severity = "low"

    # 3) Procesos “raros” (o poco comunes) -> medium (para simular falsos positivos)
    if event_type == "process":
        if process_name is None:
            severity = "medium"
            tags += ["missing_process_name"]
        elif process_name not in COMMON_PROCESSES:
            severity = "medium"
            tags += ["uncommon_process"]
        elif process_name in SUSPICIOUS_BUT_BENIGN:
            tags += ["noisy_process"]  # sigue siendo benigno pero “sospechoso”

    # 4) Network: conexiones/dns normalmente low, pero algunas se vuelven noisy
    if event_type == "network" and action in {"dns_query", "connect"}:
        tags += ["network_noise"]

    # 5) Allowlist / maintenance tags (para que luego FAISS aprenda FP)
    if user in ALLOWLIST_USERS:
        tags += ["allowlisted_user"]
    if action == "process_start" and user == "admin":
        tags += ["admin_activity"]
    # Nota: si quieres maintenance_window real, lo modelamos por tiempo/host (ver abajo en el cambio)

    return severity, tags

def background_noise_events(
    rng: random.Random,
    episode_id: int,
    seed: int,
    start: datetime,
    n: int,
) -> List[Event]:
    events: List[Event] = []
        # Burst benigno: ráfaga de auth_fail (parece ataque, pero es ruido)
    if rng.random() < 0.08:  # 8% de episodios con burst
        asset = choose_asset(rng)
        user = rng.choice(USERS)
        base_dt = start + timedelta(seconds=rng.randint(1, 120))
        for k in range(8):
            dt = base_dt + timedelta(seconds=2*k)
            severity, tags = assign_noise_severity_and_tags("auth", "login_attempt", "fail", user, None)
            events.append(Event(
                timestamp=iso(dt),
                episode_id=episode_id, seed=seed,
                event_type="auth",
                host=asset.host,
                user=user,
                src_ip=asset.ip,
                dst_ip=None,
                action="login_attempt",
                outcome="fail",
                severity=severity,
                process_name="systemd",
                tags=tags + ["burst"]
            ))

    for _ in range(n):
        dt = start + timedelta(seconds=rng.randint(1, 600))
        asset = choose_asset(rng)
        user = rng.choice(USERS)

        ev_type = rng.choice(["auth", "network", "process"])
        if ev_type == "auth":
            action = rng.choice(["login_attempt", "logout"])
            outcome = rng.choice(["success", "fail"])
        elif ev_type == "network":
            action = rng.choice(["connect", "dns_query"])
            outcome = "success"
        else:
            action = rng.choice(["process_start", "process_end"])
            outcome = "success"
        
        process = rng.choice([None, "chrome.exe", "sshd", "unknown_tool", "python"])
        severity, tags = assign_noise_severity_and_tags(
            ev_type, action, outcome, user, process
        )
                
        events.append(Event(
            timestamp=iso(dt),
            episode_id=episode_id,
            seed=seed,
            event_type=ev_type,
            host=asset.host,
            user=user,
            src_ip=asset.ip,
            dst_ip=None,
            action=action,
            outcome=outcome,
            severity=severity,
            process_name=process,
            tags=tags,
        ))
    return events


def inject_scenario_events(
    rng: random.Random,
    episode_id: int,
    seed: int,
    start: datetime,
    scenario: Dict,
) -> Tuple[List[Event], Dict[str, str], Dict[str, List[str]]]:
    events: List[Event] = []

    t_inject_start = start + timedelta(seconds=60)
    t_inject_end = start + timedelta(seconds=180)

    src = choose_asset(rng, role="workstation")
    dst = choose_asset(rng, role=rng.choice(["web", "db", "jumpbox"]))
    user = rng.choice(["admin", "alice", "bob"])

    name = scenario["name"]

    if name == "auth_anomaly_valid_account":
        events.append(Event(
            timestamp=iso(t_inject_start),
            episode_id=episode_id, seed=seed,
            event_type="auth",
            host=dst.host, user=user,
            src_ip=src.ip, dst_ip=dst.ip,
            action="login_success", outcome="success",
            severity="high",
            tags=["suspicious", "auth"],
        ))
        events.append(Event(
            timestamp=iso(t_inject_start + timedelta(seconds=30)),
            episode_id=episode_id, seed=seed,
            event_type="process",
            host=dst.host, user=user,
            src_ip=dst.ip, dst_ip=None,
            action="process_start", outcome="success",
            severity="medium",
            process_name="unknown_tool",
            tags=["post_auth"],
        ))

    elif name == "auth_bruteforce_then_success":
        for k in range(6):
            events.append(Event(
                timestamp=iso(t_inject_start + timedelta(seconds=10*k)),
                episode_id=episode_id, seed=seed,
                event_type="auth",
                host=dst.host, user=user,
                src_ip=src.ip, dst_ip=dst.ip,
                action="login_attempt", outcome="fail",
                severity="medium",
                tags=["auth", "burst"],
            ))
        events.append(Event(
            timestamp=iso(t_inject_start + timedelta(seconds=65)),
            episode_id=episode_id, seed=seed,
            event_type="auth",
            host=dst.host, user=user,
            src_ip=src.ip, dst_ip=dst.ip,
            action="login_success", outcome="success",
            severity="high",
            tags=["auth", "success_after_fail"],
        ))

    elif name == "lateral_like_remote_service":
        events.append(Event(
            timestamp=iso(t_inject_start + timedelta(seconds=20)),
            episode_id=episode_id, seed=seed,
            event_type="network",
            host=src.host, user=user,
            src_ip=src.ip, dst_ip=dst.ip,
            action="connect_remote_service", outcome="success",
            severity="high",
            tags=["lateral_like"],
        ))
        events.append(Event(
            timestamp=iso(t_inject_start + timedelta(seconds=45)),
            episode_id=episode_id, seed=seed,
            event_type="auth",
            host=dst.host, user=user,
            src_ip=src.ip, dst_ip=dst.ip,
            action="remote_auth_success", outcome="success",
            severity="high",
            tags=["lateral_like", "auth"],
        ))

    injected_window = {"start": iso(t_inject_start), "end": iso(t_inject_end)}
    expected_indicators = {
        "src_ips": [src.ip],
        "dst_ips": [dst.ip],
        "hosts": [src.host, dst.host],
        "users": [user],
    }
    return events, injected_window, expected_indicators


def inject_recurrent_benign_events(
    rng: random.Random,
    episode_id: int,
    seed: int,
    start: datetime,
    profile: Dict[str, str],
) -> Tuple[List[Event], Dict[str, str], Dict[str, List[str]]]:
    src = get_asset_by_host(profile["host"])
    dst = get_asset_by_host(profile["peer_host"])
    user = profile["user"]
    process_name = profile["process_name"]
    pattern_tag = profile["pattern_tag"]
    base_tags = ["recurrent_benign", pattern_tag]

    t_inject_start = start + timedelta(seconds=75)
    events = [
        Event(
            timestamp=iso(t_inject_start),
            episode_id=episode_id,
            seed=seed,
            event_type="auth",
            host=src.host,
            user=user,
            src_ip=src.ip,
            dst_ip=dst.ip,
            action="login_success",
            outcome="success",
            severity="high",
            tags=["suspicious", *base_tags],
        ),
        Event(
            timestamp=iso(t_inject_start + timedelta(seconds=20)),
            episode_id=episode_id,
            seed=seed,
            event_type="process",
            host=src.host,
            user=user,
            src_ip=src.ip,
            dst_ip=None,
            action="process_start",
            outcome="success",
            severity="medium",
            process_name=process_name,
            tags=["post_auth", *base_tags],
        ),
        Event(
            timestamp=iso(t_inject_start + timedelta(seconds=40)),
            episode_id=episode_id,
            seed=seed,
            event_type="network",
            host=src.host,
            user=user,
            src_ip=src.ip,
            dst_ip=dst.ip,
            action="connect_remote_service",
            outcome="success",
            severity="medium",
            process_name=process_name,
            tags=["network_noise", *base_tags],
        ),
    ]
    injected_window = {
        "start": iso(t_inject_start),
        "end": iso(t_inject_start + timedelta(seconds=40)),
    }
    expected_indicators = {
        "src_ips": [src.ip],
        "dst_ips": [dst.ip],
        "hosts": [src.host, dst.host],
        "users": [user],
    }
    return events, injected_window, expected_indicators


HARD4_VARIANTS = (
    "v1_rename",
    "v2_types",
    "v3_semantic",
    "v4_sparse",
)


def _iso_to_epoch_ms(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    return int(dt.timestamp() * 1000.0)


def _sev_to_num(sev: Optional[str]) -> int:
    s = str(sev or "low").lower()
    return {"low": 1, "medium": 2, "high": 3}.get(s, 1)


def _sev_to_p(sev: Optional[str]) -> str:
    s = str(sev or "low").lower()
    return {"high": "P1", "medium": "P2", "low": "P3"}.get(s, "P3")


def _outcome_to_bool(outcome: Optional[str]) -> bool:
    s = str(outcome or "").lower()
    return s in {"success", "ok", "true", "1"}


def _event_type_semantic(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).lower()
    return {
        "auth": "authentication",
        "network": "netflow",
        "process": "proc_event",
    }.get(s, s)


def _action_semantic(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).lower()
    return {
        "login_attempt": "auth_try",
        "login_success": "auth_ok",
        "remote_auth_success": "remote_auth_ok",
        "connect_remote_service": "svc_connect",
        "process_start": "proc_spawn",
        "process_end": "proc_exit",
        "dns_query": "dns_lookup",
        "connect": "net_connect",
    }.get(s, s)


def to_backend_b_event(event: Dict[str, Any], *, variant: str, rng: random.Random) -> Dict[str, Any]:
    tags = list(event.get("tags") or [])

    if variant == "v2_types":
        return {
            "ts_epoch_ms": _iso_to_epoch_ms(event.get("timestamp")),
            "case_num": event.get("episode_id"),
            "rand_seed": event.get("seed"),
            "evt_type_name": event.get("event_type"),
            "asset_name": event.get("host"),
            "principal": event.get("user"),
            "src_addr": event.get("src_ip"),
            "dst_addr": event.get("dst_ip"),
            "op": event.get("action"),
            "ok": _outcome_to_bool(event.get("outcome")),
            "sev_level": _sev_to_num(event.get("severity")),
            "proc": event.get("process_name"),
            "tag_blob": ",".join(tags),
        }

    if variant == "v3_semantic":
        return {
            "tstamp": event.get("timestamp"),
            "incident_case": event.get("episode_id"),
            "seed_id": event.get("seed"),
            "kind": _event_type_semantic(event.get("event_type")),
            "node": event.get("host"),
            "account": event.get("user"),
            "ip_from": event.get("src_ip"),
            "ip_to": event.get("dst_ip"),
            "verb": _action_semantic(event.get("action")),
            "result_code": "OK" if _outcome_to_bool(event.get("outcome")) else "FAIL",
            "priority": _sev_to_p(event.get("severity")),
            "image": event.get("process_name"),
            "labels": [str(t).upper() for t in tags],
        }

    if variant == "v4_sparse":
        out: Dict[str, Any] = {
            "time_obs": event.get("timestamp"),
            "ep_ref": event.get("episode_id"),
            "rnd_id": event.get("seed"),
            "cat": event.get("event_type"),
            "host_ref": event.get("host"),
            "src": event.get("src_ip"),
            "dst": event.get("dst_ip"),
            "op_name_v2": event.get("action"),
            "state_text": event.get("outcome"),
            "risk": event.get("severity"),
            "tagset": tags,
            "schema_rev": "2026.03",
            "vendor": "backend_b_sparse",
        }
        if event.get("user") and rng.random() > 0.35:
            out["usr_ref"] = event.get("user")
        if event.get("process_name") and rng.random() > 0.45:
            out["proc_meta"] = {"image": event.get("process_name")}
        return out

    # v1_rename (classic backend_b drift)
    return {
        "when_utc": event.get("timestamp"),
        "case_ref": event.get("episode_id"),
        "rnd": event.get("seed"),
        "evt_kind": event.get("event_type"),
        "asset_ref": event.get("host"),
        "actor_id": event.get("user"),
        "origin_addr": event.get("src_ip"),
        "target_addr": event.get("dst_ip"),
        "op_name": event.get("action"),
        "result_state": event.get("outcome"),
        "risk_code": event.get("severity"),
        "proc_image": event.get("process_name"),
        "labels_v2": tags,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Carpeta salida")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1337)
    ap.add_argument("--noise-per-episode", type=int, default=2000,
                   help="Eventos benignos por episodio")
    ap.add_argument("--benign-rate", type=float, default=0.0,
                    help="Fraccion de episodios sin inyeccion Red Team (0.0-1.0).")
    ap.add_argument(
        "--recurrent-benign-rate",
        type=float,
        default=0.0,
        help="Fraccion de episodios benignos que inyectan un patron benigno recurrente (0.0-1.0).",
    )
    ap.add_argument(
        "--recurrent-benign-profiles",
        type=int,
        default=len(RECURRENT_BENIGN_PROFILES),
        help="Cantidad de perfiles recurrentes benignos a ciclar.",
    )
    ap.add_argument(
        "--backend-b-drift-profile",
        type=str,
        default="classic",
        choices=["classic", "hard4"],
        help="Perfil de drift de esquema para logs_backend_b.",
    )
    args = ap.parse_args()

    logs_dir = os.path.join(args.out, "logs_backend_a")
    logs_backend_b_dir = os.path.join(args.out, "logs_backend_b")
    gt_dir = os.path.join(args.out, "ground_truth")
    ensure_dir(logs_dir)
    ensure_dir(logs_backend_b_dir)
    ensure_dir(gt_dir)

    base_start = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

    drift_map: Dict[str, str] = {}
    recurrent_benign_counter = 0
    profile_count = max(1, min(int(args.recurrent_benign_profiles), len(RECURRENT_BENIGN_PROFILES)))

    for ep in range(1, args.episodes + 1):
        seed = args.base_seed + ep
        rng = random.Random(seed)

        attack_present = (rng.random() >= args.benign_rate)
        recurrent_benign = (not attack_present) and (rng.random() < args.recurrent_benign_rate)
        recurrent_profile: Optional[Dict[str, str]] = None
        if attack_present:
            scenario = SCENARIOS[(ep - 1) % len(SCENARIOS)]
        elif recurrent_benign:
            recurrent_profile = RECURRENT_BENIGN_PROFILES[recurrent_benign_counter % profile_count]
            recurrent_benign_counter += 1
            scenario = {
                "name": recurrent_profile["name"],
                "technique_ids": [],
                "description": f"Patron benigno recurrente: {recurrent_profile['pattern_tag']}.",
            }
        else:
            scenario = {
                "name": "benign",
                "technique_ids": [],
                "description": "Episodio benigno (sin inyeccion).",
            }
        t0 = base_start + timedelta(minutes=ep * 10)

        events = background_noise_events(rng, ep, seed, t0, args.noise_per_episode)
        injected_window = {"start": iso(t0), "end": iso(t0)}
        expected_indicators = {"src_ips": [], "dst_ips": [], "hosts": [], "users": []}
        if attack_present:
            injected_events, injected_window, expected_indicators = inject_scenario_events(
                rng, ep, seed, t0, scenario
            )
            events.extend(injected_events)
        elif recurrent_benign and recurrent_profile:
            injected_events, injected_window, expected_indicators = inject_recurrent_benign_events(
                rng, ep, seed, t0, recurrent_profile
            )
            events.extend(injected_events)
        events.sort(key=lambda e: e.timestamp)

        if args.backend_b_drift_profile == "hard4":
            drift_variant = HARD4_VARIANTS[(ep - 1) % len(HARD4_VARIANTS)]
        else:
            drift_variant = "v1_rename"
        drift_map[f"episode_{ep:03d}"] = drift_variant

        log_path = os.path.join(logs_dir, f"episode_{ep:03d}.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

        log_backend_b_path = os.path.join(logs_backend_b_dir, f"episode_{ep:03d}.jsonl")
        with open(log_backend_b_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(
                    json.dumps(
                        to_backend_b_event(asdict(ev), variant=drift_variant, rng=rng),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        gt = GroundTruth(
            episode_id=ep,
            seed=seed,
            attack_present=attack_present,
            scenario_name=scenario["name"],
            technique_ids=scenario["technique_ids"],
            t0=iso(t0),
            injected_window=injected_window,
            expected_indicators=expected_indicators,
        )
        gt_path = os.path.join(gt_dir, f"episode_{ep:03d}.json")
        gt_payload = asdict(gt)
        gt_payload["backend_b_drift_variant"] = drift_variant
        gt_payload["backend_b_drift_profile"] = args.backend_b_drift_profile
        gt_payload["benign_pattern_type"] = "recurrent" if recurrent_benign else "generic"
        gt_payload["benign_pattern_profile"] = recurrent_profile["name"] if recurrent_profile else None
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] episodio {ep:03d} -> {log_path} | {log_backend_b_path} | {gt_path}")

    drift_map_path = os.path.join(args.out, "backend_b_drift_map.json")
    with open(drift_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "profile": args.backend_b_drift_profile,
                "variants": list(HARD4_VARIANTS),
                "episodes": drift_map,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
