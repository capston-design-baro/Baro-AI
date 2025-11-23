import json
import time
import requests

OC = " " # openAPI 인증 ID

INPUT_FILE = "prec_2000_2025_criminal.jsonl"         # 형사 사건 목록
OUTPUT_FILE = "prec_2000_2025_criminal_full.jsonl"   # 본문까지 합친 결과

DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"


def extract_case_id(rec):
    for key in ("판례일련번호", "판례정보일련번호", "ID"):
        if key in rec and rec[key]:
            return str(rec[key])
    return None


def normalize_detail_json(data):
    if not isinstance(data, dict):
        return data

    if "PrecService" in data:
        inner = data["PrecService"]
        if isinstance(inner, dict):
            if "prec" in inner:
                detail = inner["prec"]
                if isinstance(detail, list):
                    return detail[0] if detail else {}
                return detail
        return inner

    if "prec" in data:
        detail = data["prec"]
        if isinstance(detail, list):
            return detail[0] if detail else {}
        return detail

    return data

def fetch_detail(case_id: str):
    params = {
        "OC": OC,
        "target": "prec",
        "type": "JSON",
        "ID": case_id,
    }
    resp = requests.get(DETAIL_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    return normalize_detail_json(raw)


def main():
    seen_ids = set()
    total = 0
    fetched = 0
    skipped_no_id = 0
    error_cnt = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {total} JSON decode 실패, 스킵")
                continue

            case_id = extract_case_id(meta)
            if not case_id:
                skipped_no_id += 1
                continue

            # 중복 ID 방지
            if case_id in seen_ids:
                continue
            seen_ids.add(case_id)

            try:
                detail = fetch_detail(case_id)
            except Exception as e:
                print(f"[ERROR] ID={case_id} 요청 중 에러: {e}")
                error_cnt += 1
                continue

            fetched += 1
            if fetched % 50 == 0:
                print(f"{fetched}건 상세 조회 완료 (total lines processed: {total})")

            merged = {**meta, **detail}

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

            time.sleep(0.2)

if __name__ == "__main__":
    main()
