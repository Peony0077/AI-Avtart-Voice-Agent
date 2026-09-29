"""Ingest documents into the RAG index (LanceDB, CPU embeddings).

    .venv\\Scripts\\python scripts/ingest.py                      # data/knowledge → data/index (rebuild)
    .venv\\Scripts\\python scripts/ingest.py --src tests/fixtures/knowledge --index data/index
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", "E:/AI-models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.rag.documents import SUPPORTED, chunk_document  # noqa: E402
from app.rag.retriever import Embedder  # noqa: E402


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "knowledge")
    ap.add_argument("--index", type=Path, default=cfg.repo_path(cfg.rag.index_dir))
    ap.add_argument("--table", default=cfg.rag.table)
    args = ap.parse_args()

    import lancedb
    import pyarrow as pa

    t0 = time.perf_counter()
    files = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in SUPPORTED)
    chunks = []
    for f in files:
        doc_chunks = chunk_document(f, args.src)
        chunks += doc_chunks
        print(f"  {f.relative_to(args.src)}: {len(doc_chunks)} chunks")
    if not chunks:
        raise SystemExit(f"no documents found in {args.src}")
    t1 = time.perf_counter()
    emb = Embedder(cfg.rag.embed_model)
    vectors = emb.passages([c.text for c in chunks])
    t2 = time.perf_counter()
    schema = pa.schema([
        pa.field("id", pa.string()), pa.field("text", pa.string()), pa.field("source", pa.string()),
        pa.field("title", pa.string()), pa.field("section", pa.string()), pa.field("page", pa.int32()),
        pa.field("chunk_index", pa.int32()), pa.field("vector", pa.list_(pa.float32(), emb.dim)),
    ])
    rows = [{"id": c.id, "text": c.text, "source": c.source, "title": c.title, "section": c.section or "",
             "page": c.page or 0, "chunk_index": c.chunk_index, "vector": v.astype("float32").tolist()}
            for c, v in zip(chunks, vectors, strict=True)]
    args.index.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(args.index))
    db.create_table(args.table, data=rows, schema=schema, mode="overwrite")
    print(f"indexed {len(chunks)} chunks from {len(files)} files → {args.index} "
          f"(parse {t1 - t0:.2f}s, embed {t2 - t1:.2f}s, total {time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
