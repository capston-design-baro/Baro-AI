import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from cfg import settings

INPUT_FILE = 'prec_labeled_final.jsonl'
DB_PATH = "./legal_db"
COLLECTION_NAME = "criminal_cases"
OPENAI_API_KEY = settings.OPENAI_API_KEY
# 글자 수 제한 설정
MAX_CHARS = 5000

# 2. ChromaDB 초기화
client = chromadb.PersistentClient(path=DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# 기존 컬렉션이 있으면 삭제
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef
)


# 3. 데이터 로드 및 배치 처리
BATCH_SIZE = 100
documents = [] 
metadatas = [] 
ids = []       

count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    for idx, line in enumerate(tqdm(lines, desc="Processing")):
        try:
            entry = json.loads(line)
            
            doc_id = f"db_idx_{idx}"
            
            doc_text = entry.get("search_text", "")
            if len(doc_text) > MAX_CHARS:
                doc_text = doc_text[:MAX_CHARS]

            labels_str = ", ".join(entry.get("label", []))
            
            meta = {
                "case_no": entry.get("case_no", "번호없음"), 
                "label": labels_str,
                "facts": entry.get("facts", "")[:3000], 
                "summary": entry.get("summary", "")[:2000]
            }
            
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(meta)
            
            if len(ids) >= BATCH_SIZE:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                ids = []
                documents = []
                metadatas = []
                
        except Exception as e:
            print(f"Error skipping line {idx}: {e}")
            continue

    # 남은 데이터 처리
    if ids:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

print(f"총 {collection.count()}개의 판례가 저장되었습니다.")