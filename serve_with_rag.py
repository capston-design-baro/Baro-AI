from fastapi import FastAPI
from pydantic import BaseModel

from baroai import app as base_app 
from services.legal_rag import classify_offense_with_rag, LawRetriever


class ClassifyRequest(BaseModel):
    text: str
    top_k: int | None = 3


app: FastAPI = base_app


@app.post("/classify")
def classify(req: ClassifyRequest):
    #RAG 기반 간단 죄명 분류: 국가법령정보 API 키가 없거나 네트워크가 불가한 경우에도 동작하며 가능 시 검색된 법령/판례 요약을 컨텍스트로 사용
    retriever = LawRetriever()
    res = classify_offense_with_rag(user_text=req.text, retriever=retriever, top_k=(req.top_k or 3))
    return res

