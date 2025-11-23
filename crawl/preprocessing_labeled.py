import json
import os

INPUT_FILE = 'prec_labeled_final.jsonl'    # 라벨링 완료된 원본
OUTPUT_FILE = 'prec_labeled_clean.jsonl'   # [최종] 중복 제거된 클린 파일

def main():
    if not os.path.exists(INPUT_FILE):
        return

    seen_ids = set() 
    unique_rows = [] 
    duplicate_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                entry = json.loads(line)
                doc_id = str(entry.get("id"))
                
                if doc_id in seen_ids:
                    duplicate_count += 1
                    continue
                
                seen_ids.add(doc_id)
                unique_rows.append(entry)
                
            except json.JSONDecodeError:
                continue

    # 파일로 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in unique_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()