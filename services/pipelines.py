from typing import Dict
from services.openai_client import respond
from cfg import settings
import re, json
from pathlib import Path
from typing import Dict, Optional

BASE = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE / "out" / "prompts"

UNCERTAIN_PAT = re.compile(r"(모르|기억\s*안|불확실|잘\s*모|대략|정확히\s*모|알\s*수\s*없)", re.I)

NEUTRAL_SYSTEM = (
"You are an assistant that extracts facts for a legal complaint draft. "
"Do not give legal advice. Use ONLY the user's text."
)

# 스타일 설정
USE_BULLET = False          
BULLET_TOKEN = "○ "         # 기호 바꾸고 싶으면 여기만 수정

COMPOSE_SYSTEM = (
    "You are a drafting assistant for Korean criminal complaints.\n"
    "Produce legal-style paragraphs in Korean."
)

def load_fewshot_prompt(offense: str) -> str:
    base = (PROMPT_DIR / f"{offense}_compose_prompt.txt").read_text(encoding="utf-8")
    if not USE_BULLET:
        base = base.replace("각 문단은 '○' 기호로 시작합니다.", "각 문단은 줄바꿈으로 구분합니다.")
        base = base.replace("○ ", "") 
    return base


BASE = Path(__file__).resolve().parents[1]

def classify_need_caution(text: str) -> str | None:
    prompt = (
        "사용자 서술을 검토하여, 자문이 꼭 필요한 고위험 신호가 있으면 한 줄 경고를 한국어로 출력하고, "
        "없으면 'NONE'만 출력하세요. 예) 성범죄/시효 임박/즉시 피해 방지 필요/고액 등.\n\n"
        f"[사용자 서술]\n{text}"
    )
    out = respond(settings.OPENAI_MODEL, NEUTRAL_SYSTEM, prompt)
    if not out:
        return None
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
    # 1) few-shot 프롬프트 로드
    fewshot = load_fewshot_prompt(meta.title_ko)
    # 2) 실제 작성용 USER 프롬프트
    user = (
        f"{fewshot}\n"
        #f"[죄명] {meta.title_ko} ({meta.statute_ref})\n"
        f"[사건 요소(JSON)]\n{json.dumps(collected, ensure_ascii=False)}\n\n"
        f"[증거 메모]\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "주의:\n"
        "- 항목명('누가/언제/어디서/왜/어떻게')을 노출하지 마십시오.\n"
        "- 조문/항 번호는 검색 또는 예시에 포함된 경우에만 사용하십시오(임의 생성 금지).\n"
        "- 불명확한 부분을 본문에 '□(확인 필요)'로 쓰지 말고, 누락된 요소는 추후 질문 단계에서 보완합니다.\n"
        "- 명백한 사실이 아닌 사항에 대해 단정적인 어조를 절대 사용하지 마십시오."
    )
    out = respond(settings.OPENAI_MODEL, COMPOSE_SYSTEM, user)

    # 3) 불필요한 머리말 제거
    draft = postprocess_complaint(out)

    return {"offense": meta.offense, "title": meta.title_ko, "draft": draft}

def postprocess_complaint(text: str) -> str:
    t = text.strip()
    # 헤더 라벨(후에 제거해도 됨~)
    t = t.replace("=== 범죄사실 ===", "=== 범죄사실 ===")
    t = t.replace("=== 고소이유 ===", "=== 고소이유 ===")

    if USE_BULLET:
        t = re.sub(r"^\s*[-•]\s*", BULLET_TOKEN, t, flags=re.MULTILINE)

    else:
        t = re.sub(r"^\s*[○•-]\s*", "", t, flags=re.MULTILINE)

    # 메타헤더 제거
    t = re.sub(r"^\s*(누가|언제|어디서|무엇을|왜|어떻게)\s*:\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t
