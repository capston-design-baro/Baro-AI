from fastapi import BackgroundTasks, HTTPException
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
from services.rag import run_rag_preview, map_keyword_to_offense
from services.rag_bootstrap import ensure_rag_resources

app = FastAPI(title="BARO-AI: Complaint Draft API", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 RAG DB 준비"""
    print("[STARTUP] Ensuring RAG resources...")
    ensure_rag_resources()
    print("[STARTUP] RAG resources ready!")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

SESSIONS: dict[str, dict] = {} 

class FollowupRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    history: list[dict] # [{"role":"user"|"assistant","content":"..."}]

class ComposeRequest(BaseModel):
    session_id: str

class ChatInitRequest(BaseModel):
    text: str
    offense: Optional[str] = None

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRestoreRequest(BaseModel):
    history: list[ChatHistoryItem]

@app.get("/")
def health():
    return {"ok": True, "service": app.title, "model": settings.OPENAI_CHAT_MODEL}


def _normalize_offense(offense: Optional[str]) -> Optional[str]:
    if not offense:
        return None

    normalized = offense.strip().lower()
    offense_map = {
        "fraud": "fraud",
        "사기": "fraud",
        "사기죄": "fraud",
        "cyber_fraud": "fraud",
        "사이버사기": "fraud",
        "insult": "insult",
        "모욕": "insult",
        "모욕죄": "insult",
    }

    return offense_map.get(normalized, "fraud")


def _build_session(text: str, offense: Optional[str] = None) -> dict:
    selected_offense = _normalize_offense(offense)

    return {
        "offense": selected_offense or "fraud",
        "history": [
            {"role": "user", "content": text},
        ],
        "collected": {},
        "rag_status": "pending",
        "rag_keyword": None,
        "rag_cases": [],
        "rag_error": None,
        "_selected_offense": bool(selected_offense),
    }


def _load_rag_preview(session_id: str, text: str) -> None:
    s = SESSIONS.get(session_id)
    if not s:
        return

    try:
        rag = run_rag_preview(text, k=2)
        rag_keyword = rag["keyword"]
        rag_cases = rag["cases"]

        s["rag_keyword"] = rag_keyword
        s["rag_cases"] = rag_cases
        s["rag_status"] = "ready"
        s["rag_error"] = None

        if not s.get("_selected_offense"):
            s["offense"] = map_keyword_to_offense(rag_keyword)
    except Exception as e:
        s["rag_status"] = "failed"
        s["rag_error"] = str(e)


def _progress_from_session(s: dict) -> dict:
    collected = s.get("collected") or {}
    elements = collected.get("elements") or {}
    details = collected.get("details") or {}

    return {
        "complete": False,
        "elements": elements,
        "details": details,
    }


def _send_to_session(session_id: str, message: str) -> dict:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")

    offense = s["offense"]
    meta = get_offense_meta(offense)

    s["history"].append({"role": "user", "content": message})
    user_text = _user_window(s["history"], max_chars=1800)

    # LLM 호출
    parsed = extract_all(user_text, offense)
    elements = enforce_elements(meta, parsed.get("elements", {}), user_text)
    details  = enforce_details(parsed.get("details", {}), offense)

    old_elements = s.get("collected", {}).get("elements", {})
    for eid, old_rec in old_elements.items():
        if "_asked_count" in old_rec:
            elements.setdefault(eid, {})["_asked_count"] = old_rec["_asked_count"]

    old_details = s.get("collected", {}).get("details", {})
    for did, old_rec in old_details.items():
        if "_asked_count" in old_rec:
            details.setdefault(did, {})["_asked_count"] = old_rec["_asked_count"]
    s["collected"] = {
        "elements": elements,
        "details": details,
    }
    s["details"] = details

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
        "session_id": session_id,
        "reply": reply,
        "caution": bool(caution_msg),
        "progress": {"complete": complete, "elements": elements, "details": details},
    }

@app.post("/chat/init")
def chat_init(req: ChatInitRequest, background_tasks: BackgroundTasks):  
    sid = str(uuid4())
    SESSIONS[sid] = _build_session(req.text, req.offense)
    background_tasks.add_task(_load_rag_preview, sid, req.text)

    return {
        "session_id": sid,
        "offense": SESSIONS[sid]["offense"],
        "rag_status": SESSIONS[sid]["rag_status"],
        "rag_keyword": SESSIONS[sid]["rag_keyword"], #str
        "rag_cases": SESSIONS[sid]["rag_cases"], #list[dict] {case_no, label, text}
    }


@app.get("/chat/rag/{session_id}")
def chat_rag(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")

    return {
        "session_id": session_id,
        "rag_status": s.get("rag_status", "pending"),
        "rag_keyword": s.get("rag_keyword"),
        "rag_cases": s.get("rag_cases") or [],
    }


@app.post("/chat/send")
def chat_send(req: ChatMessageRequest):
    return _send_to_session(req.session_id, req.message)


@app.post("/chat/restore")
def chat_restore(req: ChatRestoreRequest, background_tasks: BackgroundTasks):
    user_messages = [
        item.content.strip()
        for item in req.history
        if item.role == "user" and item.content and item.content.strip()
    ]

    if not user_messages:
        raise HTTPException(400, "복원할 사용자 메시지가 없습니다.")

    sid = str(uuid4())
    SESSIONS[sid] = _build_session(user_messages[0])
    background_tasks.add_task(_load_rag_preview, sid, user_messages[0])

    last_response = None
    for message in user_messages[1:]:
        last_response = _send_to_session(sid, message)

    return {
        "session_id": sid,
        "offense": SESSIONS[sid]["offense"],
        "rag_status": SESSIONS[sid]["rag_status"],
        "rag_keyword": SESSIONS[sid]["rag_keyword"],
        "rag_cases": SESSIONS[sid]["rag_cases"],
        "progress": last_response["progress"] if last_response else _progress_from_session(SESSIONS[sid]),
    }

@app.post("/chat/compose")
def chat_compose(req: ComposeRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")
    meta = get_offense_meta(s["offense"])

    sections_payload = compose_complaint(meta=meta, collected=s.get("collected", {}), evidence=[])
    return sections_payload
