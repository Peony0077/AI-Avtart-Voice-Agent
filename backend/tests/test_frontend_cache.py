"""The page is re-checked on every load (a cached copy ran old code); hashed assets are kept."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _Frontend


def test_page_no_cache_assets_immutable(tmp_path: Path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "assets" / "index-abc123.js").write_text("x")
    app = FastAPI()
    app.mount("/", _Frontend(directory=tmp_path, html=True))
    c = TestClient(app)
    assert c.get("/").headers["cache-control"] == "no-cache"
    assert c.get("/index.html").headers["cache-control"] == "no-cache"
    assert "immutable" in c.get("/assets/index-abc123.js").headers["cache-control"]
