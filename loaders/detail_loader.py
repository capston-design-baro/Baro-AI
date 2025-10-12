# loaders/detail_loader.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import yaml

# offenses YAML이 있는 디렉토리 기준 경로
BASE = Path(__file__).resolve().parents[1]
OFFENSE_DIR = BASE / "data" / "offenses"  # 예: data/offenses/fraud.yaml

@lru_cache(maxsize=16)
def get_detail_schema(offense: str) -> list[dict]:
    """
    offenses/{offense}.yaml 파일 안의 `details:` 섹션을 읽어
    [{id,label,must,nice,questions}] 형태로 정규화해 반환.
    """
    path = OFFENSE_DIR / f"{offense}.yaml"
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    details = doc.get("details") or []
    norm = []
    for d in details:
        slots = d.get("slots", {}) or {}
        norm.append({
            "id": d["id"],
            "label": d.get("label", d["id"]),
            "must": list(slots.get("must", []) or []),
            "nice": list(slots.get("nice_to_have", []) or []),
            "questions": d.get("questions", {}) or {},
        })
    return norm
