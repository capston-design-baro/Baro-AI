from fastapi import HTTPException
from uuid import uuid4
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from cfg import settings
from loaders.offense_loader import get_offense_meta
from services.pipelines import classify_need_caution, extract_elements, generate_followup, compose_complaint, triage_check

app = FastAPI(title="BARO-AI: Complaint Draft API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 메모리 세션 저장소 (배포하면 Redis/DB로 확장해야할듯)
SESSIONS: dict[str, dict] = {} 
# 구조 예시:
# {
#   "session_id": {
#       "offense": "fraud",
#       "history": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
#       "collected": {},               # extract_elemnets 결과 누적
#       "party_info_idx": 0,           # (선택) 공통질문 인덱스 -> data/mixins에서 확인
#   }
# }

class StartRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    text: str

class FollowupRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    history: list[dict] # [{"role":"user"|"assistant","content":"..."}]

class ComposeRequest(BaseModel):
    offense: Literal["fraud", "insult"]
    collected: dict
    evidence_notes: list[str] = []

class ChatInitRequest(BaseModel):
    offense: Literal["fraud", "insult"]

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

# baroai.py
class TriageSelect(BaseModel):
    session_id: str
    option_key: str  # "continue_fraud" | "switch_civil_notice" | "switch_civil_complaint"

@app.post("/chat/triage") #민사 내용증명 내용은 할지 말지 논의
def chat_triage(req: TriageSelect):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션 없음")

    if req.option_key == "continue_fraud":
        return {"ok": True, "message": "사기 고소 초안을 계속 진행합니다."}

    if req.option_key == "switch_civil_notice":
        s["offense"] = "civil_notice"   # 내부 오프펜스 키 전환
        # 민사용 YAML 로드(data/offenses/civil_notice.yaml)
        return {"ok": True, "message": "민사 내용증명 초안 모드로 전환했습니다."}

    if req.option_key == "switch_civil_complaint":
        s["offense"] = "civil_loan"     # 대여금 반환 청구
        return {"ok": True, "message": "민사 대여금 반환 청구 초안 모드로 전환했습니다."}

    raise HTTPException(400, "알 수 없는 옵션")

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
        "message": "사건 개요를 자유롭게 적어주세요. (언제, 어디서, 누구와, 어떤 일인지)"
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

    # ✅ dead-end 감지
    triage = triage_check(meta.offense, user_text)

    # 3) 누락 요소에 대한 다음 질문 선택
    follow = generate_followup(extracted, meta)

    # triage가 발생했고 아직 follow가 있다면 → follow 대신 triage 제시(루프 중단)
    if triage:
        return {
        "session_id": req.session_id,
        "reply": triage["advisory"],
        "triage": {
            "reason": "deception_missing",
            "options": triage["options"]
        },
        "progress": {
            "complete": False,
            "elements": extracted
        }
        }

    # 4) 어시스턴트 응답 결정
    if follow:
        reply = follow["question"]
        complete = False
    else:
        reply = "필요한 정보가 어느 정도 모였어요. 작성완료라고 입력해주시면 초안을 만들어 드릴게요."
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

#임시로 대충 적어놓음 -> 실제 고소장 기반으로 퀄리티 끌어올릴 예정
@app.post("/chat/compose")
def chat_compose(req: ChatMessageRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "세션을 찾을 수 없습니다. /chat/init 먼저 호출하세요.")
    meta = get_offense_meta(s["offense"])

    # 필요하면 메시지 내용이 '작성 완료'인지 체크해서 history에 남겨도 됨
    if req.message:
        s["history"].append({"role": "user", "content": req.message})

    draft = compose_complaint(meta=meta, collected=s.get("collected", {}), evidence=[])
    return draft
