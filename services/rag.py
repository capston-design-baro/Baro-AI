from __future__ import annotations
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from cfg import settings

BASE = Path(__file__).resolve().parents[1]

OPENAI_API_KEY = settings.OPENAI_API_KEY

DB_PATH = BASE / "data" / "rag_db"
CSV_PATH = BASE / "data" / "crime_mapping_worklist.csv"

COLLECTION_NAME = "criminal_cases"
LLM_MODEL = settings.OPENAI_COMPOSE_MODEL 

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def load_offense_names(csv_path: str):
    offense_names: list[str] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    for line in lines[1:]:
        if not line.strip():
            continue
        name = line.split(",", 1)[0].strip()
        if name:
            offense_names.append(name)

    return offense_names


MAJOR_OFFENSES = load_offense_names(str(CSV_PATH))
OFFENSE_LIST_STR = ", ".join(MAJOR_OFFENSES)


# 사건 개요 → 죄목 키워드 
def predict_crime_keyword(user_query: str) -> str:
    prompt = f"""
    너는 형사 사건 고소장을 분류하는 보조 도구야.

    [규칙]
    - 아래 '죄목 리스트'에 있는 항목들 중에서 **가장 관련성이 높은 죄목 하나만** 골라서 출력해라.
    - 애매하더라도, 리스트 중에서 가장 가까운 죄목을 하나 선택해라.
    - '기타', '모름' 같은 단어는 절대 쓰지 말고, 죄목 이름만 딱 한 단어(또는 한 구절)로 출력해라.

    [죄목 리스트]
    {OFFENSE_LIST_STR}

    [사용자 사건 개요]
    {user_query}
    """

    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


# 벡터 검색 + 키워드 필터
def search_similar_cases_filtered(
    query_text: str, keyword: str, k: int = 3
) -> list[dict]:
    """Chroma에서 유사 판례를 가져오고, label 안에 keyword가 들어간 것만 필터."""
    results = collection.query(
        query_texts=[query_text],
        n_results=50,
    )

    filtered_cases: list[dict] = []

    # 결과가 전혀 없을 수도 있으니 방어
    if not results or not results.get("ids"):
        return []

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta = metadatas[i]
        label = meta.get("label", "") or ""

        if keyword not in label:
            continue

        case_info = {
            "case_no": meta.get("case_no", ""),
            "label": label,
            "summary": meta.get("summary", ""),
            "facts": meta.get("facts", ""),
            "score": distances[i],
        }
        filtered_cases.append(case_info)

        if len(filtered_cases) >= k:
            break

    return filtered_cases


# 판례 요약 문자열 생성
def summarize_cases_for_ui(user_text: str, cases: list[dict]) -> list[dict]:
    if not cases:
        return []

    cases_block = ""
    for idx, c in enumerate(cases, start=1):
        cases_block += (
            f"[판례 {idx}]\n"
            f"- 사건번호: {c.get('case_no', '')}\n"
            f"- 죄목(label): {c.get('label', '')}\n"
            f"- 사실관계(facts): {c.get('facts', '')[:800]}\n\n"
        )

    system_prompt = """
    너는 형사 사건 판례 검색 엔진의 결과 요약 봇이야.
    감정적인 공감이나 서론, 결론, 면책 문구(법적 효력이 없다는 경고 등)를 절대 붙이지 마.
    사용자가 원하는 포맷에 맞춰서 필요한 정보만 기계적으로 채워.
    """

    user_prompt = f"""
    [사용자 사건 개요]
    {user_text}

    [후보 판례들]
    {cases_block}

    위 정보를 참고해서, 아래 형식의 JSON만 출력해라.

    {{
      "cases": [
        {{
          "case_no": "<해당 판례 사건번호 그대로>",
          "label": "<해당 판례 label 그대로>",
          "text": "(1) [사건번호: xxxx] (유사도 높음)\\n- 사건 요약: ...\\n- 결과: ...\\n- 내 사건과의 공통점: ..."
        }},
        ...
      ]
    }}

    주의:
    - JSON 이외의 텍스트는 출력하지 마라.
    - "text" 안의 줄바꿈은 반드시 \\n 으로 표현해라.
    - 각 판례는 반드시 하나씩 매칭해서 작성하고, 판례 순서는 입력 순서를 그대로 유지해라.
    """

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
        out_cases = data.get("cases", [])
        normalized: list[dict] = []
        for fallback, c_out in zip(cases, out_cases):
            normalized.append(
                {
                    "case_no": c_out.get("case_no") or fallback.get("case_no", ""),
                    "label": c_out.get("label") or fallback.get("label", ""),
                    "text": c_out.get("text", "").replace("\\n", "\n").strip()
                    or f"(판례 {fallback.get('case_no', '')}) 요약을 불러오지 못했습니다.",
                }
            )
        return normalized

    # LLM이 JSON을 제대로 안 주거나 에러날 때, 최소한 fallback 문자열 생성
    except Exception:
        fallback_list: list[dict] = []
        for idx, c in enumerate(cases, start=1):
            txt = (
                f"({idx}) [사건번호: {c.get('case_no','')}] (유사도 높음)\n"
                f"- 사건 요약: {c.get('facts','')[:200]}\n"
                f"- 결과: (판결 요지는 판례 원문을 참고해야 합니다.)\n"
                f"- 내 사건과의 공통점: (사실관계를 바탕으로 유사점을 검토해야 합니다.)"
            )
            fallback_list.append(
                {
                    "case_no": c.get("case_no", ""),
                    "label": c.get("label", ""),
                    "text": txt,
                }
            )
        return fallback_list


# offense 매핑
def map_keyword_to_offense(keyword: str) -> str:
    k = (keyword or "").strip()

    # 임시로 지정
    if "모욕" in k or "명예훼손" in k or "명예 훼손" in k or "정보통신망" in k:
        return "insult"

    # 일단 나머지는 전부 fraud 템플릿 사용
    return "fraud"

def run_rag_preview(user_text: str, k: int = 2) -> dict:
    """
    - user_text 기반으로 CSV 죄목 리스트에서 keyword 추정
    - Chroma에서 유사 판례 k개 검색
    - 각 판례에 대해 UI에 바로 쓸 수 있는 요약 문자열 생성

    반환 예시:
    {
      "keyword": "사기",
      "cases": [
        { "case_no": "...", "label": "...", "text": "(1) [...] ..." },
        { "case_no": "...", "label": "...", "text": "(2) [...] ..." },
      ]
    }
    """
    keyword = predict_crime_keyword(user_text)
    raw_cases = search_similar_cases_filtered(user_text, keyword, k=k)
    summarized = summarize_cases_for_ui(user_text, raw_cases)

    return {
        "keyword": keyword, 
        "cases": summarized, 
    }
