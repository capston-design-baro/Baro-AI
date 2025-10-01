from typing import Dict
from services.openai_client import respond
from cfg import settings
import re, json
from pathlib import Path
import yaml


NEUTRAL_SYSTEM = (
"You are an assistant that extracts facts for a legal complaint draft. "
"Do not give legal advice. Use ONLY the user's text."
)

BASE = Path(__file__).resolve().parents[1]

def triage_check(offense: str, user_text: str) -> dict | None:
    path = BASE / f"data/triage/{offense}_rules.yaml"
    if not path.exists():
        return None
    rules = yaml.safe_load(open(path, "r", encoding="utf-8"))
    t = user_text.replace("\n", " ").lower()

    def hit(patterns):
        cnt = 0
        for p in patterns:
            if re.search(p, t, re.I):
                cnt += 1
        return cnt

    strong = hit(rules.get("dead_end_signals", {}).get("strong", []))
    weak   = hit(rules.get("dead_end_signals", {}).get("weak", []))
    negate = hit(rules.get("negate_if_present", []))

    th = rules.get("thresholds", {})
    cond = (strong >= th.get("strong_min", 1)
            and weak >= th.get("weak_min", 1)
            and negate <= th.get("negate_max", 0))

    if cond:
        return {
            "advisory": rules.get("advisory_text", ""),
            "options":  rules.get("options", [])
        }
    return None

def classify_need_caution(text: str) -> str | None:
    prompt = (
        "사용자 서술을 검토하여, 자문이 꼭 필요한 고위험 신호가 있으면 한 줄 경고를 한국어로 출력하고, "
        "없으면 'NONE'만 출력하세요. 예) 성범죄/시효 임박/즉시 피해 방지 필요/고액 등.\n\n"
        f"[사용자 서술]\n{text}"
    )
    out = respond(settings.OPENAI_MODEL, NEUTRAL_SYSTEM, prompt)
    return (
        "이 사건의 경우 법률 전문가과의 상담을 권장합니다."
        if out.strip().upper() != "NONE"
        else None
    )

def extract_elements(text: str, meta) -> Dict[str, dict]:
    element_list = [f"- {e.id}:{e.label}" for e in meta.elements]
    prompt = (
        "다음 구성요건 목록에 대해, 사용자의 서술이 각 요소를 'satisfied|missing|unclear' 중 무엇으로 볼지와, 1~2문장 요약을 JSON으로 만들어주세요. "
        "키는 요소 id를 사용.\n\n"
        f"[요소]\n{chr(10).join(element_list)}\n\n[사용자 서술]\n{text}"
    )
    out = respond(settings.OPENAI_MODEL, NEUTRAL_SYSTEM, prompt)
    try:
        import json
        parsed = json.loads(out)
        return parsed
    except Exception:
        return {e.id: {"status": "unclear", "summary": ""} for e in meta.elements}

def generate_followup(extracted: Dict[str, dict], meta) -> dict | None:
    # missing/unclear 중 하나를 골라 해당 요소의 첫 질문을 반환
    target = None
    for e in meta.elements:
        status = extracted.get(e.id, {}).get("status", "unclear")
        if status in ("missing", "unclear"):
            target = e
            break
    if not target:
        return None
    q = target.questions[0].text if target.questions else f"{target.label}에 대해 더 알려주세요."
    return {"element": target.id, "question": q}

def compose_complaint(meta, collected: dict, evidence: list[str]):
    user = (
        "아래의 구조화 데이터와 증거 메모를 바탕으로, 고소취지/범죄사실/고소이유 형태의 초안 텍스트를 한국어로 작성.\n"
        "법률 자문·전략·결론은 금지. 불명확한 부분은 '□(확인 필요)'로 표기.\n\n"
        f"[죄명]{meta.title_ko} ({meta.statute_ref})\n"
        f"[구조화 데이터]{collected}\n[증거 메모]{evidence}\n"
    )
    out = respond(settings.OPENAI_MODEL, NEUTRAL_SYSTEM, user)
    return {"offense": meta.offense, "title": meta.title_ko, "draft": out}
