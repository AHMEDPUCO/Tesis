#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

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

# --- Reglas simples para variar severidad/tags en ruido benigno ---

SERVICE_ACCOUNTS = {"svc_backup"}
SCANNER_IPS = {"10.0.10.99"}          # IPs benignas que generan ruido (escáner interno)
MAINTENANCE_HOSTS = {"web-01"}        # hosts con mantenimiento frecuente
ALLOWLIST_USERS = {"svc_backup"}      # allowlist simple por usuario

COMMON_PROCESSES = {"chrome.exe", "systemd", "sshd", "python"}
SUSPICIOUS_BUT_BENIGN = {"python"}    # ejemplo: genera FP en algunos contextos (opcional)

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
                host=asset["host"],
                user=user,
                src_ip=asset["ip"],
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Carpeta salida")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1337)
    ap.add_argument("--noise-per-episode", type=int, default=2000,
                   help="Eventos benignos por episodio")
    args = ap.parse_args()

    logs_dir = os.path.join(args.out, "logs_backend_a")
    gt_dir = os.path.join(args.out, "ground_truth")
    ensure_dir(logs_dir)
    ensure_dir(gt_dir)

    base_start = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

    for ep in range(1, args.episodes + 1):
        seed = args.base_seed + ep
        rng = random.Random(seed)

        scenario = SCENARIOS[(ep - 1) % len(SCENARIOS)]
        t0 = base_start + timedelta(minutes=ep * 10)

        events = background_noise_events(rng, ep, seed, t0, args.noise_per_episode)
        injected_events, injected_window, expected_indicators = inject_scenario_events(
            rng, ep, seed, t0, scenario
        )
        events.extend(injected_events)
        events.sort(key=lambda e: e.timestamp)

        log_path = os.path.join(logs_dir, f"episode_{ep:03d}.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

        gt = GroundTruth(
            episode_id=ep,
            seed=seed,
            scenario_name=scenario["name"],
            technique_ids=scenario["technique_ids"],
            t0=iso(t0),
            injected_window=injected_window,
            expected_indicators=expected_indicators,
        )
        gt_path = os.path.join(gt_dir, f"episode_{ep:03d}.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(asdict(gt), f, ensure_ascii=False, indent=2)

        print(f"[OK] episodio {ep:03d} -> {log_path} | {gt_path}")


if __name__ == "__main__":
    main()
