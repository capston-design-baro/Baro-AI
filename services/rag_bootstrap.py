from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import chromadb
import requests
from chromadb.utils import embedding_functions

from cfg import settings

# 기본 경로와 상수 정의
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
RAG_DB_PATH = DATA_DIR / "rag_db"
DATASET_PATH = BASE / "crawl" / "prec_labeled_final.jsonl"
COLLECTION_NAME = "criminal_cases"

MAX_CHARS = 5000
BATCH_SIZE = 100


def ensure_rag_resources() -> None:
    """
    RAG DB가 준비돼 있는지 확인.
    - S3에서 chroma.sqlite3 다운로드 (없을 때만)
    """
    ensure_rag_db_from_s3()


def ensure_rag_db_from_s3() -> None:
    """
    S3에서 미리 생성된 chroma.sqlite3을 다운로드.
    이미 존재하면 스킵.
    """
    sqlite_path = RAG_DB_PATH / "chroma.sqlite3"

    if sqlite_path.exists():
        print("[RAG] chroma.sqlite3 already exists. Skipping download.")
        return

    url = settings.RAG_DB_URL
    if not url:
        raise RuntimeError(
            "RAG_DB_URL is not configured. Please set it in .env file."
        )

    print(f"[RAG] Downloading chroma.sqlite3 from S3...")
    RAG_DB_PATH.mkdir(parents=True, exist_ok=True)
    download_file(url, sqlite_path)
    print(f"[RAG] Download complete: {sqlite_path}")


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    """대용량 presigned 파일을 스트리밍 다운로드 후 임시 파일에서 교체."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    tmp_path = destination.with_suffix(destination.suffix + ".download")
    downloaded_mb = 0

    with tmp_path.open("wb") as tmp_file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                tmp_file.write(chunk)
                downloaded_mb += len(chunk) / (1024 * 1024)
                if int(downloaded_mb) % 50 == 0 and int(downloaded_mb) > 0:
                    print(f"[RAG] Downloaded {int(downloaded_mb)}MB...")

    tmp_path.replace(destination)


# ========================================
# 개발자 전용: 로컬에서 임베딩 생성
# ========================================

def build_rag_db(dataset_path: Path) -> None:
    """
    [개발자 전용] 데이터셋을 임베딩해 Chroma 컬렉션을 최초 생성.
    사용자는 이 함수를 호출하지 않음. S3에서 미리 생성된 DB를 다운로드함.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set before building the RAG DB.")

    client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    processed = 0

    with dataset_path.open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_text = (entry.get("search_text") or "")[:MAX_CHARS]
            if not doc_text:
                continue

            labels = entry.get("label") or []
            labels_iter: Iterable[str] = [labels] if isinstance(labels, str) else labels

            metadata = {
                "case_no": entry.get("case_no", "번호없음"),
                "label": ", ".join(labels_iter),
                "facts": (entry.get("facts") or "")[:3000],
                "summary": (entry.get("summary") or "")[:2000],
            }

            doc_id = f"db_idx_{idx}"
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(metadata)
            processed += 1

            if len(ids) >= BATCH_SIZE:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                documents.clear()
                metadatas.clear()
                ids.clear()

            if processed and processed % 500 == 0:
                print(f"[RAG] Embedded {processed} records...")

    if ids:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"[RAG] Build complete. Stored {collection.count()} cases.")


__all__ = [
    "ensure_rag_resources",
    "ensure_rag_db_from_s3",
    "build_rag_db",
]
