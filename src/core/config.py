from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class Asset:
    host: str
    ip: str
    role: str
    criticality: str
# Definición de activos en el entorno de ciberseguridad
ASSETS: List[Asset] = [
    Asset("ws-01",  "10.0.10.21", "workstation", "low"),
    Asset("ws-02",  "10.0.10.22", "workstation", "low"),
    Asset("web-01", "10.0.20.10", "web",         "medium"),
    Asset("db-01",  "10.0.20.20", "db",          "high"),
    Asset("jump-01","10.0.10.5",  "jumpbox",     "medium"),
]
# Definición de usuarios en el entorno de ciberseguridad
USERS = ["alice", "bob", "svc_backup", "admin"]
