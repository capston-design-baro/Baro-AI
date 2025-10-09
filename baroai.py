from fastapi import HTTPException
from uuid import uuid4
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from cfg import settings
from loaders.offense_loader import get_offense_meta
from services.pipelines import classify_need_caution, extract_elements, generate_followup, compose_complaint

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
    return {"ok": True, "service": app.title, "model": settings.OPENAI_MODEL}

@app.post("/chat/init")
def chat_init(req: ChatInitRequest):
    # 죄목 메타 로드 확인 
    _ = get_offense_meta(req.offense)

    sid = str(uuid4())
    SESSIONS[sid] = {
        "offense": req.offense,
        "history": [],
        "collected": {},
    }
    return {
        "session_id": sid,
        "message": "사건 개요를 자유롭게 최대한 자세히 시간 순으로 적어주세요. (언제, 어디서, 누구와, 어떤 일인지)"
    }

@app.post("/chat/send")
def chat_send(req: ChatMessageRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")

    offense = s["offense"]
    meta = get_offense_meta(offense)

    # 1) 사용자 메시지 기록 -> 회의에서 말한데로 일단 세션으로 받음
    s["history"].append({"role": "user", "content": req.message})

    # 2) 지금까지의 사용자 발화만 합쳐서 요소 추출
    user_text = "\n".join(m["content"] for m in s["history"] if m["role"] == "user")
    extracted = extract_elements(user_text, meta) or {}
    s["collected"] = extracted  # 누적 저장

    # 3) 누락 요소에 대한 다음 질문 선택
    follow = generate_followup(extracted, meta)

    # 4) 어시스턴트 응답 결정
    if follow:
        reply = follow["question"]
        complete = False
    else:
        reply = "필요한 정보가 어느 정도 모였어요."
        complete = True

    s["history"].append({"role": "assistant", "content": reply})
    
    caution_msg = classify_need_caution(user_text)
    
    return {
        "session_id": req.session_id,
        "reply": reply,
        "caution": bool(caution_msg),
        "progress": {
            "complete": complete,
            "elements": extracted  # 진행률 표시 용
        }
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
