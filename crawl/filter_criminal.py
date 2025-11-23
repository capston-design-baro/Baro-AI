import json

INPUT_FILE = "prec_2000_2025.jsonl"            # 전체 판례 데이터
OUTPUT_FILE = "prec_2000_2025_criminal.jsonl"  # 형사 사건만 저장할 파일

def filter_criminal_cases(input_path: str, output_path: str):
    total = 0
    kept = 0
    skipped_no_key = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {total} JSON 디코딩 실패: {e}")
                continue

            case_type = rec.get("사건종류명")

            if case_type is None:
                skipped_no_key += 1
                continue

            # 형사 사건만 필터
            if case_type == "형사":
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

if __name__ == "__main__":
    filter_criminal_cases(INPUT_FILE, OUTPUT_FILE)
