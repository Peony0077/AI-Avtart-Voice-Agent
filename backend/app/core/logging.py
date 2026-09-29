"""Logging + per-turn metrics sink (JSON lines)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from loguru import logger

_metrics_lock = threading.Lock()
_metrics_path: Path | None = None


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    global _metrics_path
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:7}</level> | {message}")
    logger.add(log_dir / "backend.log", level="DEBUG", rotation="20 MB", retention=5, encoding="utf-8", enqueue=True)
    _metrics_path = log_dir / "metrics.jsonl"


def write_metrics(record: dict) -> None:
    if _metrics_path is None:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _metrics_lock, _metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
