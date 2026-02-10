from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


@dataclass
class MemoryHit:
    score: float
    case: Dict[str, Any]


class FaissMemory:
    """
    Memoria vectorial persistente:
    - data/memory/cases.jsonl  (casos legibles)
    - data/memory/index.faiss  (índice FAISS)
    Cosine similarity vía Inner Product con embeddings normalizados.
    """

    def __init__(
        self,
        dir_path: str = "data/memory",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.dir_path = dir_path
        self.cases_path = os.path.join(dir_path, "cases.jsonl")
        self.index_path = os.path.join(dir_path, "index.faiss")
        os.makedirs(self.dir_path, exist_ok=True)

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        self.cases: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(self.dim)

        self._load()

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embs = embs.astype("float32")
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return _l2_normalize(embs)

    def _load(self) -> None:
        # casos
        self.cases = []
        if os.path.exists(self.cases_path):
            with open(self.cases_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.cases.append(json.loads(line))

        # índice
        if os.path.exists(self.index_path) and len(self.cases) > 0:
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            if len(self.cases) > 0:
                embs = self._embed([c["text"] for c in self.cases])
                self.index.add(embs)
            self._persist_index()

    def _persist_index(self) -> None:
        faiss.write_index(self.index, self.index_path)

    def add_case(
        self,
        *,
        text: str,
        label: str,        # "TP" | "FP" | "UNCERTAIN"
        decision: str,     # "block_ip" | "no_block" | "escalate"
        reason: str,
        tags: Optional[List[str]] = None,
        source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        case_id = (self.cases[-1]["case_id"] + 1) if self.cases else 1
        case = {
            "case_id": case_id,
            "created_at": _iso_now(),
            "text": text,
            "label": label,
            "decision": decision,
            "reason": reason,
            "tags": tags or [],
            "source": source or {},
        }

        with open(self.cases_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

        self.cases.append(case)
        self.index.add(self._embed([text]))
        self._persist_index()
        return case

    def search(self, *, text: str, k: int = 3, threshold: float = 0.75) -> List[MemoryHit]:
        if not self.cases:
            return []

        q = self._embed([text])
        scores, idxs = self.index.search(q, k)

        hits: List[MemoryHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            s = float(score)
            if s < threshold:
                continue
            hits.append(MemoryHit(score=s, case=self.cases[int(idx)]))
        return hits
