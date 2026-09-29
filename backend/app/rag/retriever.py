"""CPU embeddings (bge-small-en-v1.5) + LanceDB vector store + grounded context building."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from app.core.config import RAGCfg

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval instruction


class Embedder:
    def __init__(self, model: str) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(2, (torch.get_num_threads() or 4) // 2))
        self.model = SentenceTransformer(model, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)[0]


@dataclass
class RetrievalResult:
    context: str | None
    citations: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg: RAGCfg, index_dir: Path) -> None:
        import lancedb

        self.cfg = cfg
        self.embedder = Embedder(cfg.embed_model)
        self.db = lancedb.connect(str(index_dir))
        names = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        names = getattr(names, "tables", names)  # newer lancedb returns a paged response object
        self.table = self.db.open_table(cfg.table) if cfg.table in list(names) else None
        rows = self.table.count_rows() if self.table is not None else 0
        logger.info(f"RAG ready: {rows} chunks in {index_dir}")
        self.rows = rows

    def retrieve(self, question: str) -> RetrievalResult:
        if self.table is None or not question.strip():
            return RetrievalResult(None, [], {"chunks": 0})
        t0 = time.perf_counter()
        vec = self.embedder.query(question)
        t1 = time.perf_counter()
        query = self.table.search(vec)
        query = query.distance_type("cosine") if hasattr(query, "distance_type") else query.metric("cosine")
        hits = query.limit(self.cfg.top_k).to_list()
        t2 = time.perf_counter()
        for h in hits:
            h["score"] = round(1.0 - float(h["_distance"]), 3)
        good = [h for h in hits if h["score"] >= self.cfg.min_score]
        stats = {"embed_ms": round((t1 - t0) * 1000, 1), "search_ms": round((t2 - t1) * 1000, 1),
                 "top_score": hits[0]["score"] if hits else None, "used": len(good)}
        if not good:
            if hits and hits[0]["score"] >= self.cfg.weak_score:
                # related material exists but nothing is relevant enough: tell the model explicitly
                return RetrievalResult("(No passage in the user's documents answers this question.)", [], stats)
            return RetrievalResult(None, [], stats)
        parts, citations, used = [], [], 0
        for i, h in enumerate(good, 1):
            text = h["text"]
            if used + len(text) > self.cfg.max_context_chars:
                text = text[: max(0, self.cfg.max_context_chars - used)]
            if not text:
                break
            used += len(text)
            parts.append(f"[{i}] ({h['title']}{' - ' + h['section'] if h['section'] else ''}) {text}")
            citations.append({"n": i, "source": h["source"], "title": h["title"], "section": h["section"],
                              "page": h["page"] if h["page"] and h["page"] > 0 else None, "score": h["score"],
                              "snippet": h["text"][:220]})
        return RetrievalResult("\n".join(parts), citations, stats)
