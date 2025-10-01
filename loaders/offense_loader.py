from pathlib import Path
import yaml
from schemas.offense import Offense, Question

BASE = Path(__file__).resolve().parents[1]
_cache = {}

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_mixin(name: str) -> list[Question]:
    mpath = BASE / f"data/mixins/{name}.yaml"
    if not mpath.exists():
        print(f"[WARN] mixin 파일 없음: {mpath}")
        return []
    m = _load_yaml(mpath)
    if m.get("mixin") != name:
        raise RuntimeError(f"mixin 이름 불일치: {name}")
    return [Question.model_validate(q) for q in m.get("questions", [])]

def _validate_slots_cover_questions(data: dict):
    # must slot 커버되는지? 간단 검증
    for el in data.get("elements", []):
        must = set(((el.get("slots") or {}).get("must") or []))
        qslots = {q.get("slot") for q in el.get("questions", []) if q.get("slot")}
        missing = must - qslots
        if missing:
            raise RuntimeError(f"[{el.get('id')}] must 슬롯을 채우는 질문이 없습니다: {missing}")

def get_offense_meta(offense_key: str) -> Offense:
    if offense_key in _cache:
        return _cache[offense_key]

    path = BASE / f"data/offenses/{offense_key}.yaml"
    data = _load_yaml(path)
    data["templates"] = data.get("templates") or {}

    # includes mixin 병합
    merged_party = []
    for inc in data.get("includes", []):
        merged_party.extend(_load_mixin(inc))
    data["party_info"] = [q.model_dump() for q in merged_party]

    _validate_slots_cover_questions(data)  # slot-q 매핑 사전 검증

    meta = Offense.model_validate(data)
    _cache[offense_key] = meta
    return meta
