from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as url_error
from urllib import parse, request


CANONICAL_FIELDS: List[str] = [
    "timestamp",
    "episode_id",
    "seed",
    "event_type",
    "host",
    "user",
    "src_ip",
    "dst_ip",
    "action",
    "outcome",
    "severity",
    "process_name",
    "tags",
]


CRITICAL_FIELDS: List[str] = [
    "timestamp",
    "event_type",
    "src_ip",
    "host",
    "action",
    "outcome",
    "severity",
    "tags",
]


USEFUL_PARTIAL_REQUIRED_FIELDS = {
    "timestamp",
    "event_type",
    "src_ip",
    "severity",
    "tags",
}


USEFUL_PARTIAL_MIN_COVERAGE = 0.75


FALLBACK_ALIASES: Dict[str, List[str]] = {
    # Keep fallback strict so dynamic mode actually needs LLM under schema drift.
    "timestamp": ["timestamp"],
    "episode_id": ["episode_id"],
    "seed": ["seed"],
    "event_type": ["event_type"],
    "host": ["host"],
    "user": ["user"],
    "src_ip": ["src_ip"],
    "dst_ip": ["dst_ip"],
    "action": ["action"],
    "outcome": ["outcome"],
    "severity": ["severity"],
    "process_name": ["process_name"],
    "tags": ["tags"],
}


@dataclass
class MappingResult:
    mapping: Dict[str, str]
    confidence: float
    source: str
    signature: str
    error: Optional[str] = None
    cache_hit: bool = False
    llm_called: bool = False


class DynamicSchemaMapper:
    def __init__(
        self,
        *,
        enabled: bool,
        cache_path: str,
        api_key: Optional[str],
        provider: str = "gemini",
        model: str = "gemini-1.5-flash",
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen3:8b",
        min_confidence: float = 0.75,
        llm_timeout_sec: float = 8.0,
        shared_cache_path: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.cache_path = cache_path
        self.shared_cache_path = shared_cache_path or ""
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.min_confidence = min_confidence
        self.llm_timeout_sec = max(1.0, float(llm_timeout_sec))
        self._cache: Dict[str, Any] = self._load_cache()
        self._shared_cache: Dict[str, Any] = self._load_shared_cache()

    def _cache_set(
        self,
        cache_key: str,
        mapping: Dict[str, str],
        confidence: float,
        source: str,
        *,
        write_shared: bool = False,
    ) -> None:
        payload = {
            "mapping": dict(mapping),
            "confidence": float(confidence),
            "source": str(source),
        }
        self._cache[cache_key] = payload
        self._save_cache()
        if write_shared and self.shared_cache_path:
            self._shared_cache[cache_key] = payload
            self._save_shared_cache()

    def infer_mapping(
        self,
        *,
        backend: str,
        sample_events: List[Dict[str, Any]],
        contract_hints: Optional[Dict[str, str]] = None,
    ) -> MappingResult:
        contract_hints = dict(contract_hints or {})
        signature = self._schema_signature(sample_events, contract_hints=contract_hints)
        key_signature = self._schema_signature(sample_events, contract_hints=None)
        if not sample_events:
            return MappingResult(mapping={}, confidence=0.0, source="empty", signature=signature)

        cache_key = f"{backend}:{signature}"
        relaxed_cache_key = f"{backend}:{key_signature}"
        cached_result = self._cache_lookup(cache_key=cache_key, relaxed_cache_key=relaxed_cache_key, signature=signature)
        if cached_result is not None:
            return cached_result

        fallback_mapping = build_fallback_mapping(sample_events)
        base_mapping = dict(contract_hints)
        # Prefer exact observed aliases when they exist, but keep contract hints available.
        base_mapping.update(fallback_mapping)
        # Si el fallback ya cubre todo el esquema canonico, no necesita LLM.
        if len(base_mapping) == len(CANONICAL_FIELDS):
            result = MappingResult(
                mapping=base_mapping,
                confidence=1.0,
                source="fallback_full_alias",
                signature=signature,
                cache_hit=False,
                llm_called=False,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result

        if not self.enabled:
            result = MappingResult(
                mapping=base_mapping,
                confidence=0.0,
                source="fallback_disabled",
                signature=signature,
                cache_hit=False,
                llm_called=False,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result

        llm_result: Optional[MappingResult]
        if self.provider == "gemini":
            if not self.api_key:
                result = MappingResult(
                    mapping=base_mapping,
                    confidence=0.0,
                    source="fallback_missing_api_key",
                    signature=signature,
                    cache_hit=False,
                    llm_called=False,
                )
                self._cache_store_variants(
                    cache_key=cache_key,
                    relaxed_cache_key=relaxed_cache_key,
                    mapping=result.mapping,
                    confidence=result.confidence,
                    source=result.source,
                )
                return result
            llm_result = self._infer_with_gemini(
                backend=backend,
                sample_events=sample_events,
                signature=signature,
                contract_hints=contract_hints,
            )
        elif self.provider == "ollama":
            llm_result = self._infer_with_ollama(
                backend=backend,
                sample_events=sample_events,
                signature=signature,
                contract_hints=contract_hints,
            )
        else:
            result = MappingResult(
                mapping=fallback_mapping,
                confidence=0.0,
                source=f"fallback_unknown_provider_{self.provider}",
                signature=signature,
                cache_hit=False,
                llm_called=False,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result
        if llm_result is None:
            result = MappingResult(
                mapping=base_mapping,
                confidence=0.0,
                source="fallback_llm_error",
                signature=signature,
                error="llm_result_none",
                cache_hit=False,
                llm_called=True,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result

        if llm_result.error:
            result = MappingResult(
                mapping=base_mapping,
                confidence=float(llm_result.confidence or 0.0),
                source=_fallback_error_source(llm_result.source),
                signature=signature,
                error=llm_result.error,
                cache_hit=False,
                llm_called=True,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result

        merged = dict(base_mapping)
        merged.update(llm_result.mapping)

        if llm_result.confidence < self.min_confidence:
            if _is_useful_partial_mapping(merged):
                result = MappingResult(
                    mapping=merged,
                    confidence=max(float(llm_result.confidence), _critical_coverage(merged)),
                    source=self.provider,
                    signature=signature,
                    cache_hit=False,
                    llm_called=True,
                )
                self._cache_store_variants(
                    cache_key=cache_key,
                    relaxed_cache_key=relaxed_cache_key,
                    mapping=result.mapping,
                    confidence=result.confidence,
                    source=result.source,
                )
                return result
            result = MappingResult(
                mapping=base_mapping,
                confidence=llm_result.confidence,
                source="fallback_low_confidence",
                signature=signature,
                cache_hit=False,
                llm_called=True,
            )
            self._cache_store_variants(
                cache_key=cache_key,
                relaxed_cache_key=relaxed_cache_key,
                mapping=result.mapping,
                confidence=result.confidence,
                source=result.source,
            )
            return result

        result = MappingResult(
            mapping=merged,
            confidence=llm_result.confidence,
            source=self.provider,
            signature=signature,
            cache_hit=False,
            llm_called=True,
        )
        self._cache_store_variants(
            cache_key=cache_key,
            relaxed_cache_key=relaxed_cache_key,
            mapping=result.mapping,
            confidence=result.confidence,
            source=result.source,
        )
        return result

    def _cache_lookup(
        self,
        *,
        cache_key: str,
        relaxed_cache_key: str,
        signature: str,
    ) -> Optional[MappingResult]:
        for key, lookup_type in (
            (cache_key, "exact"),
            (relaxed_cache_key, "relaxed"),
        ):
            cached = self._cache.get(key)
            result = self._mapping_result_from_cache(cached, signature=signature, lookup_type=lookup_type)
            if result is not None:
                return result
            shared = self._shared_cache.get(key)
            result = self._mapping_result_from_cache(shared, signature=signature, lookup_type=f"shared_{lookup_type}")
            if result is not None:
                return result
        return None

    def _mapping_result_from_cache(
        self,
        cached: Any,
        *,
        signature: str,
        lookup_type: str,
    ) -> Optional[MappingResult]:
        if (
            not isinstance(cached, dict)
            or not isinstance(cached.get("mapping"), dict)
            or _should_retry_cached_source(str(cached.get("source") or ""))
        ):
            return None
        cached_source = str(cached.get("source") or "cache")
        source = cached_source if lookup_type == "exact" else f"{cached_source}_cache_{lookup_type}"
        return MappingResult(
            mapping={str(k): str(v) for k, v in (cached.get("mapping") or {}).items()},
            confidence=float(cached.get("confidence") or 1.0),
            source=source,
            signature=signature,
            cache_hit=True,
            llm_called=False,
        )

    def _cache_store_variants(
        self,
        *,
        cache_key: str,
        relaxed_cache_key: str,
        mapping: Dict[str, str],
        confidence: float,
        source: str,
    ) -> None:
        write_shared = _is_shared_reusable_source(source, mapping, confidence, self.min_confidence)
        self._cache_set(cache_key, mapping, confidence, source, write_shared=write_shared)
        if _is_relaxed_reusable_source(source, mapping, confidence, self.min_confidence):
            self._cache_set(relaxed_cache_key, mapping, confidence, source, write_shared=write_shared)

    def _infer_with_gemini(
        self,
        *,
        backend: str,
        sample_events: List[Dict[str, Any]],
        signature: str,
        contract_hints: Dict[str, str],
    ) -> Optional[MappingResult]:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        url = f"{endpoint}?{parse.urlencode({'key': self.api_key})}"
        prompt = _build_mapping_prompt(
            backend=backend,
            sample_events=sample_events,
            contract_hints=contract_hints,
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        try:
            req = request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.llm_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except TimeoutError as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="gemini_error_timeout",
                signature=signature,
                error=str(exc),
                cache_hit=False,
                llm_called=True,
            )
        except (url_error.URLError, ValueError) as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="gemini_error_transport",
                signature=signature,
                error=str(exc),
                cache_hit=False,
                llm_called=True,
            )

        try:
            data = json.loads(body)
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < 0 or end <= start:
                return MappingResult(
                    mapping={},
                    confidence=0.0,
                    source="gemini_error_parse",
                    signature=signature,
                    error=f"missing_json_object: {_preview_text(text)}",
                    cache_hit=False,
                    llm_called=True,
                )
            parsed = json.loads(text[start : end + 1])
            mapping = parsed.get("mapping") or {}
            if not isinstance(mapping, dict):
                return MappingResult(
                    mapping={},
                    confidence=0.0,
                    source="gemini_error_parse",
                    signature=signature,
                    error=f"mapping_not_dict: {_preview_text(text)}",
                    cache_hit=False,
                    llm_called=True,
                )
            cleaned = _sanitize_mapping(mapping, sample_events)
            raw_conf = parsed.get("confidence")
            if raw_conf is None:
                raw_conf = _coverage_confidence(cleaned)
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
            return MappingResult(
                mapping=cleaned,
                confidence=confidence,
                source="gemini_raw",
                signature=signature,
                cache_hit=False,
                llm_called=True,
            )
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="gemini_error_parse",
                signature=signature,
                error=f"{type(exc).__name__}: {exc}",
                cache_hit=False,
                llm_called=True,
            )

    def _infer_with_ollama(
        self,
        *,
        backend: str,
        sample_events: List[Dict[str, Any]],
        signature: str,
        contract_hints: Dict[str, str],
    ) -> Optional[MappingResult]:
        url = f"{self.ollama_url}/api/generate"
        prompt = _build_mapping_prompt(
            backend=backend,
            sample_events=sample_events,
            contract_hints=contract_hints,
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            req = request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.llm_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except TimeoutError as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="ollama_error_timeout",
                signature=signature,
                error=str(exc),
                cache_hit=False,
                llm_called=True,
            )
        except (url_error.URLError, ValueError) as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="ollama_error_transport",
                signature=signature,
                error=str(exc),
                cache_hit=False,
                llm_called=True,
            )

        try:
            data = json.loads(body)
            text = str(data.get("response", "")).strip()
            parsed = _extract_json_obj(text)
            if parsed is None:
                return MappingResult(
                    mapping={},
                    confidence=0.0,
                    source="ollama_error_parse",
                    signature=signature,
                    error=f"response_not_json: {_preview_text(text)}",
                    cache_hit=False,
                    llm_called=True,
                )
            mapping = parsed.get("mapping") or parsed
            if not isinstance(mapping, dict):
                return MappingResult(
                    mapping={},
                    confidence=0.0,
                    source="ollama_error_parse",
                    signature=signature,
                    error=f"mapping_not_dict: {_preview_text(text)}",
                    cache_hit=False,
                    llm_called=True,
                )
            cleaned = _sanitize_mapping(mapping, sample_events)
            raw_conf = parsed.get("confidence")
            if raw_conf is None or float(raw_conf) <= 0.0:
                raw_conf = _coverage_confidence(cleaned)
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
            return MappingResult(
                mapping=cleaned,
                confidence=confidence,
                source="ollama_raw",
                signature=signature,
                cache_hit=False,
                llm_called=True,
            )
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            return MappingResult(
                mapping={},
                confidence=0.0,
                source="ollama_error_parse",
                signature=signature,
                error=f"{type(exc).__name__}: {exc}",
                cache_hit=False,
                llm_called=True,
            )

    def _schema_signature(
        self,
        sample_events: List[Dict[str, Any]],
        *,
        contract_hints: Optional[Dict[str, str]] = None,
    ) -> str:
        keys = sorted(_collect_keys(sample_events[:8]))
        hints = contract_hints or {}
        hint_bits = [f"{key}={hints[key]}" for key in sorted(hints)]
        raw = "|".join(keys + ["#"] + hint_bits)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path:
            return {}
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _load_shared_cache(self) -> Dict[str, Any]:
        if not self.shared_cache_path:
            return {}
        if not os.path.exists(self.shared_cache_path):
            return {}
        try:
            with open(self.shared_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _save_shared_cache(self) -> None:
        if not self.shared_cache_path:
            return
        cache_dir = os.path.dirname(self.shared_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.shared_cache_path, "w", encoding="utf-8") as f:
            json.dump(self._shared_cache, f, ensure_ascii=False, indent=2)


def build_fallback_mapping(sample_events: List[Dict[str, Any]]) -> Dict[str, str]:
    keys = _collect_keys(sample_events[:8])
    mapping: Dict[str, str] = {}
    for canonical, aliases in FALLBACK_ALIASES.items():
        for alias in aliases:
            if alias in keys:
                mapping[canonical] = alias
                break
    return mapping


def _sanitize_mapping(mapping: Dict[str, Any], sample_events: List[Dict[str, Any]]) -> Dict[str, str]:
    keys = _collect_keys(sample_events[:8])
    cleaned: Dict[str, str] = {}
    for canonical, source in mapping.items():
        c = str(canonical)
        s = str(source)
        if c not in CANONICAL_FIELDS:
            # Allow inverted format: {"source_field":"canonical_field"}
            if s in CANONICAL_FIELDS and c in keys:
                cleaned[s] = c
            continue
        if s in keys:
            cleaned[c] = s
    return cleaned


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _coverage_confidence(cleaned_mapping: Dict[str, str]) -> float:
    if not cleaned_mapping:
        return 0.0
    return len(cleaned_mapping) / float(len(CANONICAL_FIELDS))


def _critical_coverage(mapping: Dict[str, str]) -> float:
    if not mapping:
        return 0.0
    covered = sum(1 for field in CRITICAL_FIELDS if field in mapping)
    return covered / float(len(CRITICAL_FIELDS))


def _is_useful_partial_mapping(mapping: Dict[str, str]) -> bool:
    if not USEFUL_PARTIAL_REQUIRED_FIELDS.issubset(mapping.keys()):
        return False
    return _critical_coverage(mapping) >= USEFUL_PARTIAL_MIN_COVERAGE


def _should_retry_cached_source(source: str) -> bool:
    return source.startswith("fallback_llm_error")


def _is_relaxed_reusable_source(
    source: str,
    mapping: Dict[str, str],
    confidence: float,
    min_confidence: float,
) -> bool:
    if source == "fallback_full_alias":
        return True
    if source in {"gemini", "ollama"}:
        return bool(mapping) and (confidence >= min_confidence or _is_useful_partial_mapping(mapping))
    return False


def _is_shared_reusable_source(
    source: str,
    mapping: Dict[str, str],
    confidence: float,
    min_confidence: float,
) -> bool:
    if source == "fallback_full_alias":
        return True
    if source in {"gemini", "ollama"}:
        return bool(mapping) and (confidence >= min_confidence or _is_useful_partial_mapping(mapping))
    return False


def _fallback_error_source(source: str) -> str:
    if source.startswith("ollama_error_timeout") or source.startswith("gemini_error_timeout"):
        return "fallback_llm_error_timeout"
    if source.startswith("ollama_error_transport") or source.startswith("gemini_error_transport"):
        return "fallback_llm_error_transport"
    if source.startswith("ollama_error_parse") or source.startswith("gemini_error_parse"):
        return "fallback_llm_error_parse"
    return "fallback_llm_error"


def _preview_text(text: str, limit: int = 240) -> str:
    raw = str(text or "").replace("\n", "\\n").replace("\r", "\\r").strip()
    if len(raw) > limit:
        return raw[: limit - 3] + "..."
    return raw


def _collect_keys(events: List[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for event in events:
        if isinstance(event, dict):
            _collect_event_keys(event, prefix="", depth=0, out=keys)
    return keys


def _collect_event_keys(value: Any, *, prefix: str, depth: int, out: set[str]) -> None:
    if not isinstance(value, dict):
        return
    if depth > 2:
        return
    for key, sub in value.items():
        key_s = str(key)
        path = f"{prefix}.{key_s}" if prefix else key_s
        out.add(path)
        if isinstance(sub, dict):
            _collect_event_keys(sub, prefix=path, depth=depth + 1, out=out)


def _build_mapping_prompt(
    *,
    backend: str,
    sample_events: List[Dict[str, Any]],
    contract_hints: Dict[str, str],
) -> str:
    sample = sample_events[:3]
    observed = _summarize_observed_fields(sample_events[:8], max_fields=32)
    hints_text = json.dumps(contract_hints, ensure_ascii=False, sort_keys=True) if contract_hints else "{}"
    sample_text = json.dumps(sample, ensure_ascii=False)
    canonical_semantics = (
        "- timestamp: event time, usually ISO-8601 or epoch milliseconds\n"
        "- episode_id: case or episode identifier\n"
        "- seed: random seed or run seed identifier\n"
        "- event_type: coarse event category such as auth, network, process\n"
        "- host: asset, host, node, endpoint or hostname\n"
        "- user: principal, actor, account or username\n"
        "- src_ip: origin/source IP address\n"
        "- dst_ip: destination/target IP address\n"
        "- action: operation verb such as login_success, login_attempt, connect_remote_service, process_start\n"
        "- outcome: result such as success/fail, ok/error, true/false\n"
        "- severity: risk or severity level such as low/medium/high or numeric priority\n"
        "- process_name: process image, executable name or process path\n"
        "- tags: labels, tag arrays, tag blobs, categories or rule labels"
    )
    required_priority = [
        "timestamp",
        "event_type",
        "src_ip",
        "host",
        "action",
        "outcome",
        "severity",
        "tags",
    ]
    return (
        "You are mapping backend log fields to a canonical security-event schema.\n"
        f"Backend: {backend}\n"
        f"Canonical fields: {CANONICAL_FIELDS}\n"
        f"High-priority canonical fields: {required_priority}\n"
        "Canonical field semantics:\n"
        f"{canonical_semantics}\n"
        "Known tool-contract hints (canonical -> backend alias candidates). Use these when they match the observed data:\n"
        f"{hints_text}\n"
        "Observed flattened keys with types and example values:\n"
        f"{observed}\n"
        "Sample backend events:\n"
        f"{sample_text}\n"
        "Rules:\n"
        "1. Return ONLY one strict JSON object with shape {\"mapping\":{\"canonical_field\":\"source_field\"},\"confidence\":0.0}.\n"
        "2. Only use source_field values that appear in the observed flattened keys list.\n"
        "3. Prefer the most semantically correct field, not just the most similar name.\n"
        "4. Prefer contract hints when they are consistent with the examples.\n"
        "5. If a canonical field is unclear, omit it rather than guessing.\n"
        "6. Confidence must be in [0,1] and reflect how complete and reliable the mapping is.\n"
        "7. Do not include explanations, markdown or extra keys."
    )


def _summarize_observed_fields(events: List[Dict[str, Any]], *, max_fields: int = 32) -> str:
    profiles: Dict[str, Dict[str, Any]] = {}
    for event in events:
        _collect_field_profiles(event, prefix="", depth=0, out=profiles)
    lines: List[str] = []
    for key in sorted(profiles)[:max_fields]:
        profile = profiles[key]
        types = ", ".join(sorted(profile["types"])) or "unknown"
        examples = "; ".join(profile["examples"]) or "n/a"
        lines.append(f"- {key} | types={types} | examples={examples}")
    return "\n".join(lines) if lines else "- <no observed keys>"


def _collect_field_profiles(
    value: Any,
    *,
    prefix: str,
    depth: int,
    out: Dict[str, Dict[str, Any]],
) -> None:
    if not isinstance(value, dict) or depth > 2:
        return
    for key, sub in value.items():
        key_s = str(key)
        path = f"{prefix}.{key_s}" if prefix else key_s
        profile = out.setdefault(path, {"types": set(), "examples": []})
        profile["types"].add(_value_type_name(sub))
        example = _format_value_example(sub)
        if example and example not in profile["examples"] and len(profile["examples"]) < 2:
            profile["examples"].append(example)
        if isinstance(sub, dict):
            _collect_field_profiles(sub, prefix=path, depth=depth + 1, out=out)


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _format_value_example(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())[:4]
        return "{keys=" + ",".join(keys) + "}"
    if isinstance(value, list):
        parts = [str(v) for v in value[:3]]
        return "[" + ",".join(parts) + "]"
    text = str(value).strip()
    if len(text) > 48:
        text = text[:45] + "..."
    return text
