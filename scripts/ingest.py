"""Ingest documents into the RAG index (LanceDB, CPU embeddings).

    .venv\\Scripts\\python scripts/ingest.py                      # data/knowledge → data/index (rebuild)
    .venv\\Scripts\\python scripts/ingest.py --src tests/fixtures/knowledge --index data/index
    .venv\\Scripts\\python scripts/ingest.py --url https://example.com/page   # save into <src>/web/, then index all
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from model_home import hf_home  # noqa: E402  (scripts/model_home.py: the env, .env, else the platform default)

os.environ.setdefault("HF_HOME", hf_home())
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.rag.documents import SUPPORTED, chunk_document  # noqa: E402
from app.rag.retriever import Embedder  # noqa: E402


def fetch(url: str, dest_dir: Path, max_bytes: int = 5_000_000) -> Path:
    """Save a web page as HTML under dest_dir (the file name keeps host and path, shown in citations)."""
    import httpx

    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(f"not a web URL: {url}")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{u.netloc}{u.path}".strip("/"))[:120] or u.netloc
    r = httpx.get(url, follow_redirects=True, timeout=30.0,
                  headers={"User-Agent": "AI-Avatar-Voice-Agent ingest"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        raise SystemExit(f"{url}: not an HTML/text page ({ctype})")
    if len(r.content) > max_bytes:
        raise SystemExit(f"{url}: page larger than {max_bytes} bytes")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{slug}.html"
    path.write_bytes(r.content)
    print(f"  saved {url} -> {path.name} ({len(r.content) // 1024} KB)")
    return path


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "knowledge")
    ap.add_argument("--index", type=Path, default=cfg.repo_path(cfg.rag.index_dir))
    ap.add_argument("--table", default=cfg.rag.table)
    ap.add_argument("--url", action="append", default=[], help="web page to add (repeatable); saved to <src>/web/")
    args = ap.parse_args()
    for url in args.url:
        fetch(url, args.src / "web")

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
    print(f"indexed {len(chunks)} chunks from {len(files)} files -> {args.index} "
          f"(parse {t1 - t0:.2f}s, embed {t2 - t1:.2f}s, total {time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
