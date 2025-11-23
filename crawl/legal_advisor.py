import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from cfg import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY
DB_PATH = "./legal_db"
COLLECTION_NAME = "criminal_cases"
LLM_MODEL = "gpt-5" 

# 설정 및 연결
chroma_client = chromadb.PersistentClient(path=DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME, 
    embedding_function=openai_ef
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def load_offense_names(csv_path):
    offense_names = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    for line in lines[1:]:
        if not line.strip():
            continue
        name = line.split(",", 1)[0].strip()
        if name:
            offense_names.append(name)

    return offense_names

# 사용 예시
CSV_PATH = "crime_mapping_worklist.csv" 
MAJOR_OFFENSES = load_offense_names(CSV_PATH)

# 범죄 유형 먼저 판단하기
OFFENSE_LIST_STR = ", ".join(MAJOR_OFFENSES)

def predict_crime_keyword(user_query):
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
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# 필터링 검색 (Two-Step Search)
# 1차로 벡터 검색을 하되, Python 레벨에서 키워드 필터링
def search_similar_cases_filtered(query_text, keyword, k=3):
    
    results = collection.query(
        query_texts=[query_text],
        n_results=50 
    )
    
    filtered_cases = []
    
    for i in range(len(results['ids'][0])):
        label = results['metadatas'][0][i]['label']
        
        if keyword in label or keyword == '기타':
            case_info = {
                "case_no": results['metadatas'][0][i]['case_no'],
                "label": label,
                "summary": results['metadatas'][0][i]['summary'],
                "facts": results['metadatas'][0][i]['facts'],
                "score": results['distances'][0][i]
            }
            filtered_cases.append(case_info)
            
        if len(filtered_cases) >= k:
            break
            
    if not filtered_cases:
        for i in range(min(k, len(results['ids'][0]))):
            filtered_cases.append({
                "case_no": results['metadatas'][0][i]['case_no'],
                "label": results['metadatas'][0][i]['label'],
                "summary": results['metadatas'][0][i]['summary'],
                "facts": results['metadatas'][0][i]['facts'],
                "score": results['distances'][0][i]
            })
            
    return filtered_cases

# 생성 함수
def generate_legal_advice(user_query, cases):
    context_text = ""
    for idx, case in enumerate(cases):
        context_text += f"""
        [판례 {idx+1}] 사건번호: {case['case_no']} / 죄목: {case['label']}
        사실관계 요약: {case['facts'][:800]}
        --------------------------------------------------
        """

    system_prompt = """
    너는 형사 사건 판례 검색 엔진의 결과 요약 봇이야.
    감정적인 공감이나 서론, 결론, 사족(법적 효력이 없다는 경고 등)을 절대 붙이지 마.
    사용자가 원하는 포맷에 맞춰서 기계적으로 정보를 채워.
    """

    user_prompt = f"""
    [사용자 사건]
    "{user_query}"
    
    [검색된 유사 판례]
    {context_text}
    
    위 판례들을 참고하여 아래 포맷으로만 출력해.
    
    === 출력 포맷 ===
    
    ## 1. 추정 죄목
    - [죄목명]: [이 사건에 적용될 죄목 정의 및 이유 1줄 요약]
    
    ## 2. 유사 판례 비교
    
    (1) [판례번호 넣기] (유사도 높음)
    - 사건 요약: [판례의 사실관계 1~2문장 요약]
    - 결과: [유죄/무죄 여부 및 적용 죄목]
    - 내 사건과의 공통점: [사용자 사건과 이 판례가 왜 비슷한지 1문장 설명]

    2) [판례번호 넣기]
    - 사건 요약: [판례의 사실관계 1~2문장 요약]
    - 결과: [유죄/무죄 여부 및 적용 죄목]
    - 내 사건과의 공통점: [사용자 사건과 이 판례가 왜 비슷한지 1문장 설명]
    """

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

def main():
    print("[판례 기반 죄목 추정기] (v2.0 필터링 적용)")
    print("사건 내용을 입력하세요 (종료: exit)\n")
    
    while True:
        user_input = input("사건 개요: ")
        if user_input.lower() in ['exit', 'quit']: break
        if len(user_input) < 5: continue
        
        # 1. 키워드 예측
        predicted_keyword = predict_crime_keyword(user_input)
        
        # 2. 필터링 검색
        similar_cases = search_similar_cases_filtered(user_input, predicted_keyword, k=2) # 2개만 보여주기
        
        if not similar_cases:
            print("   -> 유사한 판례를 찾지 못했습니다.")
            continue
            
        # 3. 결과 생성
        print("   -> 분석 결과 생성 중...\n")
        advice = generate_legal_advice(user_input, similar_cases)
        
        print(advice)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()