import json
import re
import os


INPUT_JSONL = 'prec_2000_2025_criminal_full.jsonl'
OUTPUT_JSONL = 'prec_labeled_final.jsonl'

# 텍스트 refinement
def remove_html_tags(text):
    if not text: return ""
    clean = re.sub(r'<.*?>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

# 죄목 parsing 
def clean_crime_name(crime_name):
    cleaned = re.sub(r'\([^)]*\)', '', crime_name)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    cleaned = cleaned.strip()
    remove_terms = ['의 무죄 부분', '일부인정된죄명', '의 일부', '방조', '미수', '예비', '제1적죄명', '제2적죄명']
    for term in remove_terms:
        cleaned = cleaned.replace(term, '')
    return cleaned.strip()

def split_smart_crime_name(complex_name):
    split_pattern = r'\s*(?:[·,ㆍ]|(?<!@)및)\s*(?![^(]*\))'
    parts = re.split(split_pattern, complex_name)
    final_crimes = []
    for part in parts:
        cleaned = clean_crime_name(part)
        if cleaned and len(cleaned) >= 2:
            final_crimes.append(cleaned)
    return list(set(final_crimes))

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"({INPUT_JSONL})을 찾을 수 없습니다.")
        return
    
    success_count = 0
    no_label_count = 0
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as fin, \
        open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line: continue
            
            try:
                entry = json.loads(line)
                new_doc_id = f"doc_{idx}"
                
                # 1) 라벨링
                labels = []
                if "사건명" in entry and entry["사건명"]:
                    labels = split_smart_crime_name(entry["사건명"])
                
                if not labels:
                    no_label_count += 1
                    continue
                
                # 2) 텍스트 정제
                raw_summary = entry.get("판결요지", "")
                raw_content = entry.get("판례내용", "")
                clean_summary = remove_html_tags(raw_summary)
                clean_content = remove_html_tags(raw_content) 
                
                # 3) 검색 텍스트 생성
                label_str = ", ".join(labels)
                search_text = f"죄목: {label_str} | 요지: {clean_summary} | 내용: {clean_content}"
                
                # 4) 최종 저장 (ID 필드에 new_doc_id 사용)
                final_entry = {
                    "id": new_doc_id,             
                    "original_id": entry.get("id"),
                    "case_no": entry.get("사건번호"),
                    "label": labels,               
                    "summary": clean_summary,      
                    "facts": clean_content,        
                    "search_text": search_text,    
                    "ref_law": remove_html_tags(entry.get("참조조문", "")) 
                }
                
                fout.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
                success_count += 1
                    
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    main()