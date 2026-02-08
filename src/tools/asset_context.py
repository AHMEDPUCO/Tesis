from __future__ import annotations
from typing import Any, Dict, Optional
import json

# Si estás usando ASSETS como lista de dicts:
# from core.config import ASSETS
# Si estás usando dataclass Asset, adapta abajo con asset.host, asset.ip, etc.

from src.core.config import ASSETS, ASSET_METADATA

def get_asset_context(host_or_ip: str) -> Dict[str, Any]:
    """
    Tool: get_asset_context(host_or_ip)
    Retorna contexto de un asset para decisión del Blue Team.
    """
    # Normaliza
    key = (host_or_ip or "").strip()

    found = None
    # ASSETS puede ser lista de dicts o dataclasses; soportamos ambos
    for a in ASSETS:
        host = a["host"] if isinstance(a, dict) else a.host
        ip = a["ip"] if isinstance(a, dict) else a.ip
        if key == host or key == ip:
            found = a
            break

    if not found:
        return {
            "found": False,
            "query": key,
            "asset": None,
            "notes": ["asset_not_found"]
        }

    # Construir respuesta consistente
    host = found["host"] if isinstance(found, dict) else found.host
    ip = found["ip"] if isinstance(found, dict) else found.ip
    role = found["role"] if isinstance(found, dict) else found.role
    criticality = found["criticality"] if isinstance(found, dict) else found.criticality

    return {
        "found": True,
        "query": key,
        "asset": {
            "host": host,
            "ip": ip,
            "role": role,
            "criticality": criticality,
            # Puedes ampliar luego:
            "owner": ASSET_METADATA.get(host, {}).get("owner"),
            "allowlists": {
                "users": ["svc_backup"],          # ejemplo (alineado a tus tags)
                "src_ips": ["10.0.10.99"],        # ejemplo scanner benigno
                "tags": ["maintenance_window"],   # ejemplo
            }
        },
        "notes": []
    }


if __name__ == "__main__":
    print(get_asset_context("db-01"))
    print(get_asset_context("10.0.10.21"))
    print(get_asset_context("no-existe"))
###python -m src.tools.asset_context
