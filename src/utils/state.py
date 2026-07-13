"""Persist risk/strategy state across restarts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path("data/bot_state.json")


def save_state(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    data = {
        **payload,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(p)


def load_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
