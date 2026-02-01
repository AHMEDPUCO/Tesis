from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

# Modelo de datos para eventos y ground truth en un entorno de ciberseguridad
@dataclass
class Event:
    timestamp: str
    episode_id: int
    seed: int
    event_type: str        # auth/network/process/file/dns/http...
    host: str
    user: Optional[str]
    src_ip: Optional[str]
    dst_ip: Optional[str]
    action: str            # login_attempt/login_success/connect/...
    outcome: str           # success/fail/unknown
    severity: str          # low/medium/high
    process_name: Optional[str] = None
    tags: Optional[List[str]] = None
# Modelo de datos para ground truth de escenarios
@dataclass
class GroundTruth:
    episode_id: int # Identificador del episodio
    seed: int      # Semilla utilizada
    scenario_name: str      # Nombre del escenario
    technique_ids: List[str]     # ATT&CK IDs referenciales
    t0: str                   # Inicio del escenario
    injected_window: Dict[str, str] # {'start': str, 'end': str}
    expected_indicators: Dict[str, List[str]] # {'ip': [], 'domain': [], 'file_hash': []}
