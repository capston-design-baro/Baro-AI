"""
로컬에서 임베딩을 생성하는 스크립트 (개발자가 1회만 실행)

사용법:
    python scripts/generate_embeddings.py

필요 환경변수:
    OPENAI_API_KEY: OpenAI API 키

필요 파일:
    crawl/prec_labeled_final.jsonl: 원본 데이터셋
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from services.rag_bootstrap import build_rag_db, RAG_DB_PATH, DATASET_PATH
from cfg import settings


def main():
    print("=" * 70)
    print("RAG DB 임베딩 생성 스크립트 (개발자 전용)")
    print("=" * 70)

    # OpenAI API 키 확인
    if not settings.OPENAI_API_KEY:
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print(".env 파일에 OPENAI_API_KEY를 설정하세요.")
        sys.exit(1)

    # 데이터셋 파일 확인
    if not DATASET_PATH.exists():
        print(f"❌ 오류: {DATASET_PATH} 파일이 존재하지 않습니다.")
        print("먼저 prec_labeled_final.jsonl 파일을 준비하세요.")
        sys.exit(1)

    print(f"✅ 데이터셋 파일: {DATASET_PATH}")
    print(f"✅ OpenAI API 키: {settings.OPENAI_API_KEY[:20]}...")
    print(f"✅ 출력 경로: {RAG_DB_PATH}")

    # 사용자 확인
    response = input("\n임베딩 생성을 시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업이 취소되었습니다.")
        sys.exit(0)

    # 임베딩 생성
    print("\n🚀 임베딩 생성 시작...")
    print("⏳ 약 10-20분 소요될 수 있습니다...")
    try:
        build_rag_db(DATASET_PATH)
        print("\n✅ 임베딩 생성 완료!")
        print(f"📁 생성된 파일: {RAG_DB_PATH / 'chroma.sqlite3'}")
        print("\n다음 단계:")
        print("  python scripts/upload_rag_db_to_s3.py")
    except Exception as e:
        print(f"\n❌ 임베딩 생성 실패: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("작업 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
