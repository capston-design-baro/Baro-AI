from fastapi import HTTPException
from uuid import uuid4
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from cfg import settings
from loaders.offense_loader import get_offense_meta
from services.pipelines import (
    extract_all,           # 단일 프롬프트 추출 
    enforce_elements,      # elements 보수 강제
    enforce_details,       # details 보수 강제
    pick_detail_followup,  # 디테일 우선 질문 선정
    pick_element_followup, # 구성요건 질문 선정
    classify_need_caution, # 고위험 경고 감지
    _user_window,
    compose_complaint
)
app = FastAPI(title="BARO-AI: Complaint Draft API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 메모리 세션 저장소
SESSIONS: dict[str, dict] = {} 

class StartRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    text: str

class FollowupRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    history: list[dict] # [{"role":"user"|"assistant","content":"..."}]

class ComposeRequest(BaseModel):
    session_id: str

class ChatInitRequest(BaseModel):
    offense: Literal["fraud", "insult"]

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
def health():
    return {"ok": True, "service": app.title, "model": settings.OPENAI_CHAT_MODEL}

@app.post("/chat/init")
def chat_init(req: ChatInitRequest):
    _ = get_offense_meta(req.offense)

    sid = str(uuid4())
    SESSIONS[sid] = {
        "offense": req.offense,
        "history": [],
        "collected": {},
    }
    return {
        "session_id": sid,
        "message": "사건 개요를 최대한 자세히 구체적인 시간 순으로 적어주세요. (언제, 어디서, 누구와, 어떤 일인지)\n고소를 진행하게 된 이유를 포함해주시면 좋습니다."
    }

@app.post("/chat/send")
def chat_send(req: ChatMessageRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")
    
    offense = s["offense"]  # "fraud"
    meta = get_offense_meta(offense)

    s["history"].append({"role": "user", "content": req.message})
    user_text = _user_window(s["history"], max_chars=1800)  # ✅ 윈도우만 전송

    # LLM 호출
    parsed = extract_all(user_text, offense)
    elements = enforce_elements(meta, parsed.get("elements", {}), user_text)
    details  = enforce_details(parsed.get("details", {}), offense)

    s["collected"] = elements
    s["details"]   = details

    # 질문 선택
    reply = pick_detail_followup(details, offense) or pick_element_followup(elements, meta) \
            or "필수 정보가 충족되었습니다. 고소장을 작성해드릴게요."
    complete = (reply.startswith("필수 정보가 충족"))

    # 중복 질문 방지
    last_assistant = next((m for m in reversed(s["history"]) if m["role"] == "assistant"), None)
    if last_assistant and last_assistant["content"].strip() == reply.strip() and not complete:
        alt = pick_element_followup(elements, meta) if reply == pick_detail_followup(details, offense) else pick_detail_followup(details, offense)
        if alt and alt.strip() != reply.strip():
            reply = alt

    s["history"].append({"role": "assistant", "content": reply})

    # 위험요소 정규식판별
    caution_msg = classify_need_caution("\n".join(m["content"] for m in s["history"] if m["role"]=="user"))

    return {
        "session_id": req.session_id,
        "reply": reply,
        "caution": bool(caution_msg),
        "progress": {"complete": complete, "elements": elements, "details": details},
    }

@app.post("/chat/compose")
def chat_compose(req: ComposeRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")
    meta = get_offense_meta(s["offense"])

    draft = compose_complaint(meta=meta, collected=s.get("collected", {}), evidence=[])
    print(draft.get("draft")) #확인용
    return draft
