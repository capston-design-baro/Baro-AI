from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from cfg import settings
from services.openai_client import respond
from loaders.offense_loader import get_offense_meta
from loaders.detail_loader import get_detail_schema


# ---------- Paths / Globals ----------
BASE = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE / "out" / "prompts"

# ---------- Config ----------
USE_BULLET = False
BULLET_TOKEN = "○ "  # 기호 바꾸려면 여기만 수정

NEUTRAL_SYSTEM = (
    "You are an assistant that extracts facts for a legal complaint draft. "
    "Do not give legal advice. Use ONLY the user's text."
)

COMPOSE_SYSTEM = (
    "You are a drafting assistant for Korean criminal complaints.\n"
    "Produce legal-style paragraphs in Korean."
)

# 애매 표현 감지
UNCERTAIN_PAT = re.compile(
    r"(경|쯤|무렵|대략|정도|언저리|기억\s*안|잘\s*모|불명확|대충|추정)",
    re.I,
)

#고위험 감지
RISK_PAT = re.compile(r"(미성년|성폭력|성범죄|강간|강제추행|공갈|상습|협박|보복|스토킹|흉기|고액|1천만|천만|시효|고의로|신변위협)", re.I)

# ---------- Few-shot Prompt ----------
@lru_cache(maxsize=16)
def load_fewshot_prompt(offense_title_ko: str) -> str:
    """
    out/prompts/{offense_title_ko}_compose_prompt.txt 를 읽어 문체 예시를 로드.
    """
    path = PROMPT_DIR / f"{offense_title_ko}_compose_prompt.txt"
    txt = path.read_text(encoding="utf-8")
    if not USE_BULLET:
        txt = txt.replace("각 문단은 '○' 기호로 시작합니다.", "각 문단은 줄바꿈으로 구분합니다.")
        txt = txt.replace("○ ", "")
    return txt


# ---------- Safety / Caution ----------
def classify_need_caution(text: str) -> str | None:
    if not text:
        return None
    return "이 사건의 경우 법률 전문가과의 상담을 권장합니다." if RISK_PAT.search(text) else None

# ---------- Single-prompt Extraction (elements + details) ----------
def _lines_for_elements(meta) -> List[str]:
    lines = []
    for e in meta.elements:
        s = _normalize_slots(getattr(e, "slots", None))
        must = s["must"]; nice = s["nice_to_have"]
        lines.append(f"- {e.id}:{e.label} | must={list(must)} | nice_to_have={list(nice)}")
    return lines


def _lines_for_details(offense: str) -> List[str]:
    lines = []
    for d in get_detail_schema(offense):
        lines.append(f"- {d['id']}:{d['label']} | must={list(d['must'])} | nice_to_have={list(d['nice'])}")
    return lines


def _build_single_prompt(user_text: str, meta, offense: str) -> str:
    elines = "\n".join(_lines_for_elements(meta))
    dlines = "\n".join(_lines_for_details(offense))
    return (
        "아래 '법적 구성요건(elements)'과 '디테일(details)' 스키마에 따라, 사용자의 서술을 **매우 보수적**으로 평가하세요.\n"
        "- 텍스트에 **명시**되지 않으면 추론하지 말고 missing으로 표기\n"
        "- 애매 표현(쯤/경/무렵/대략/정도/기억 안 남 등)은 해당 슬롯을 'unclear'\n"
        "- 각 항목의 status는 'satisfied|missing|unclear'\n"
        "- 각 항목의 slots 값은 'present|missing|unclear'\n"
        "- **must 슬롯 중 하나라도 present가 아니면 해당 항목 status는 반드시 'missing'**\n"
        "- 가능할 때 evidence에 짧게 한 구절만 인용(없으면 빈 문자열)\n\n"
        "JSON만 출력:\n"
        "{\n"
        '  "elements": {\n'
        '    "<element_id>": {\n'
        '      "status": "satisfied|missing|unclear",\n'
        '      "slots": {"<slot>": "present|missing|unclear", ...},\n'
        '      "evidence": "<짧은 인용 또는 빈 문자열>",\n'
        '      "summary": "1~2문장 요약"\n'
        "    }, ...\n"
        "  },\n"
        '  "details": {\n'
        '    "<detail_id>": {\n'
        '      "status": "satisfied|missing|unclear",\n'
        '      "slots": {"<slot>": "present|missing|unclear", ...},\n'
        '      "evidence": "<짧은 인용 또는 빈 문자열>",\n'
        '      "summary": "1~2문장 요약"\n'
        "    }, ...\n"
        "  }\n"
        "}\n\n"
        "[elements]\n" + elines + "\n\n" +
        "[details]\n" + dlines + "\n\n" +
        "[사용자 서술]\n" + user_text
    )


def extract_all(user_text: str, offense: str) -> Dict[str, Dict[str, dict]]:
    """
    LLM 1회 호출로 elements + details 동시 추출.
    반환: {"elements": {...}, "details": {...}}
    """
    meta = get_offense_meta(offense)
    prompt = _build_single_prompt(user_text, meta, offense)
    out = respond(settings.OPENAI_CHAT_MODEL, NEUTRAL_SYSTEM, prompt)
    try:
        parsed = json.loads(out)
    except Exception:
        parsed = {"elements": {}, "details": {}}
    if "elements" not in parsed or not isinstance(parsed["elements"], dict):
        parsed["elements"] = {}
    if "details" not in parsed or not isinstance(parsed["details"], dict):
        parsed["details"] = {}
    return parsed

# ---------- Enforcement (보수적 강제 규칙) ----------
def enforce_elements(meta, elements: Dict[str, dict], user_text: str) -> Dict[str, dict]:
    def has_uncertain(s: str) -> bool:
        return bool(UNCERTAIN_PAT.search(s or ""))

    for e in meta.elements:
        rec = elements.get(e.id, {}) or {}
        slots = rec.get("slots", {}) or {}
        status = rec.get("status", "missing")

        s = _normalize_slots(getattr(e, "slots", None))
        must = s["must"]

        if has_uncertain(rec.get("summary", "")) or has_uncertain(rec.get("evidence", "")) or has_uncertain(user_text):
            status = "unclear"

        for slot_name in must:
            if slots.get(slot_name) != "present":
                status = "missing"
                rec.setdefault("_missing_must", []).append(slot_name)

        rec["status"] = status
        rec["slots"] = slots
        elements[e.id] = rec
    return elements


def enforce_details(details: Dict[str, dict], offense: str) -> Dict[str, dict]:
    """
    - must 슬롯 중 하나라도 present가 아니면 status=missing 강제
    """
    schema = {d["id"]: d for d in get_detail_schema(offense)}
    for did, rec in (details or {}).items():
        spec = schema.get(did)
        if not spec:
            continue
        slots = rec.get("slots", {}) or {}
        status = rec.get("status", "missing")
        for s in spec["must"]:
            if slots.get(s) != "present":
                status = "missing"
                rec.setdefault("_missing_must", []).append(s)
        rec["status"] = status
        rec["slots"] = slots
        details[did] = rec
    return details

# ---------- Follow-up Question Pickers ----------
def _user_window(history: list[dict], max_chars: int = 1500) -> str:
    """최근 사용자 메시지부터 거꾸로 합쳐 max_chars를 넘지 않게 만든 윈도우."""
    buf = []
    used = 0
    for m in reversed(history):
        if m["role"] != "user":
            continue
        c = m["content"]
        if used + len(c) > max_chars and buf:
            break
        buf.append(c); used += len(c)
        if used >= max_chars:
            break
    return "\n".join(reversed(buf))

def pick_detail_followup(details: Dict[str, dict], offense: str) -> Optional[str]:
    """
    디테일 스키마 기준으로 가장 먼저 물을 질문 1개를 선택.
    우선순위: must → (모두 present면) nice_to_have
    """
    for spec in get_detail_schema(offense):
        rec = (details or {}).get(spec["id"], {}) or {}
        slots = rec.get("slots", {}) or {}
        # must 슬롯 우선
        for s in spec["must"]:
            if slots.get(s) in (None, "missing", "unclear"):
                return spec["questions"].get(s) or f"{spec['label']}의 '{s}' 정보를 알려주세요."
        # nice_to_have는 must 충족 후
        if all(slots.get(s) == "present" for s in spec["must"]):
            for s in spec["nice"]:
                if slots.get(s) in (None, "missing", "unclear"):
                    return spec["questions"].get(s) or f"{spec['label']}의 '{s}' 정보를 알려주세요."
    return None

def pick_element_followup(elements: Dict[str, dict], meta) -> Optional[str]:
    for e in meta.elements:
        rec = elements.get(e.id, {}) or {}
        slot_status = rec.get("slots", {}) or {}
        s = _normalize_slots(getattr(e, "slots", None))
        must = s["must"]

        for slot_name in must:
            if slot_status.get(slot_name) in (None, "missing", "unclear"):
                # 해당 슬롯 질문 찾기
                for q in getattr(e, "questions", []) or []:
                    if getattr(q, "slot", None) == slot_name:
                        return q.text
                return f"{e.label}의 '{slot_name}' 정보를 알려주세요."
    return None

# ---------- Backward-compat wrappers (기존 코드 호환) ----------
def extract_elements(text: str, meta) -> Dict[str, dict]:
    """
    (호환용) elements만 반환. 내부적으로 단일 프롬프트를 사용하여 elements를 강제 보수화 후 리턴.
    """
    offense = meta.offense if hasattr(meta, "offense") else "fraud"
    parsed = extract_all(text, offense)
    return enforce_elements(meta, parsed.get("elements", {}), text)

def generate_followup(extracted: Dict[str, dict], meta) -> Optional[dict]:
    """
    (호환용) elements 기반 질문 1개만 리턴.
    """
    q = pick_element_followup(extracted, meta)
    if not q:
        return None
    # 기존 반환 형태 유지
    target = None
    for e in meta.elements:
        status = extracted.get(e.id, {}).get("status", "unclear")
        if status in ("missing", "unclear"):
            target = e
            break
    if not target:
        # 그래도 없으면 간단 반환
        return {"element": "", "question": q}
    return {"element": target.id, "question": q}

# ---------- Compose (작성 전용) ----------
def compose_complaint(meta, collected: dict, evidence: List[str]):
    """
    - few-shot 문체 + 구조화 요소 JSON + 증거 메모를 입력으로 받아 범죄사실/고소이유 줄글 초안만 생성
    - follow-up 생성/누락 표시 없음 (send에서만 질문 처리)
    """
    fewshot = load_fewshot_prompt(meta.title_ko)
    user = (
        f"{fewshot}\n"
        f"[사건 요소(JSON)]\n{json.dumps(collected, ensure_ascii=False)}\n\n"
        f"[증거 메모]\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "주의:\n"
        "- 항목명('누가/언제/어디서/왜/어떻게')을 노출하지 마십시오.\n"
        "- 조문/항 번호는 검색 또는 예시에 포함된 경우에만 사용하십시오(임의 생성 금지).\n"
        "- 불명확/누락 정보는 본문에 표기하지 않습니다.\n"
        #"- 불명확한 부분은 '□(확인 필요)'로 표기하십시오."
        "- 명백한 사실이 아닌 사항에 대해 단정적인 어조를 절대 사용하지 마십시오.\n"
    )
    out = respond(settings.OPENAI_COMPOSE_MODEL, COMPOSE_SYSTEM, user)
    draft = postprocess_complaint(out)
    return {"offense": meta.offense, "title": meta.title_ko, "draft": draft}

def postprocess_complaint(text: str) -> str:
    """
    출력 후속 정리:
    - 불필요한 불릿/헤더 토큰 제거 또는 변환
    - 과도한 개행 정리
    """
    t = (text or "").strip()

    # 섹션 라벨(유지/정리용)
    t = t.replace("=== 범죄사실 ===", "=== 범죄사실 ===")
    t = t.replace("=== 고소이유 ===", "=== 고소이유 ===")

    if USE_BULLET:
        t = re.sub(r"^\s*[-•]\s*", BULLET_TOKEN, t, flags=re.MULTILINE)
    else:
        t = re.sub(r"^\s*[○•-]\s*", "", t, flags=re.MULTILINE)

    # 메타헤더 제거 (누가/언제 등)
    t = re.sub(r"^\s*(누가|언제|어디서|무엇을|왜|어떻게)\s*:\s*", "", t, flags=re.MULTILINE)

    # 과도한 개행 정리
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def _normalize_slots(slots_obj):
    """
    slots가 dict이든 pydantic BaseModel이든 안전하게
    {"must": [...], "nice_to_have": [...]} 형태로 변환.
    """
    if slots_obj is None:
        return {"must": [], "nice_to_have": []}

    # dict 형태
    if isinstance(slots_obj, dict):
        return {
            "must": list(slots_obj.get("must", []) or []),
            "nice_to_have": list(slots_obj.get("nice_to_have", []) or []),
        }

    # pydantic 모델 형태
    must = getattr(slots_obj, "must", None)
    if must is None:
        must = getattr(slots_obj, "required", [])  # 혹시 모델이 required로 정의된 경우 대비
    nice = getattr(slots_obj, "nice_to_have", None)
    if nice is None:
        nice = getattr(slots_obj, "optional", [])  # 혹시 optional로 정의된 경우 대비

    return {
        "must": list(must or []),
        "nice_to_have": list(nice or []),
    }