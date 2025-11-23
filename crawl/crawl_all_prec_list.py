import requests
import json
import time
from cfg import settings

BASE_URL = "http://www.law.go.kr/DRF/lawSearch.do"
OC = settings.OC # 오픈API ID

START_YEAR = 2000
END_YEAR = 2025

DISPLAY = 100    
MAX_PAGES_PER_RANGE = 1000 

def fetch_year(year: int, file_obj):
    start = f"{year}0101"
    end = f"{year}1231"
    page = 1

    print(f"\n===== Fetching year {year} ({start} ~ {end}) =====")

    while page <= MAX_PAGES_PER_RANGE:
        params = {
            "OC": OC,
            "target": "prec",
            "type": "JSON",
            "display": DISPLAY,
            "page": page,
            "prncYd": f"{start}~{end}", 
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
        except requests.RequestException as e:
            print(f"[{year}] page {page} error: {e}")
            break

        if resp.status_code != 200:
            print(f"[{year}] page {page} HTTP {resp.status_code} 에러, 이 연도는 중단")
            break

        try:
            data = resp.json()
        except ValueError:
            print(f"[{year}] page {page} JSON 디코딩 실패. 일부 내용: {resp.text[:200]}")
            break

        prec_list = None

        if isinstance(data, dict):
            if "PrecSearch" in data and isinstance(data["PrecSearch"], dict):
                if "prec" in data["PrecSearch"]:
                    prec_list = data["PrecSearch"]["prec"]
            if prec_list is None and "prec" in data:
                prec_list = data["prec"]

            if prec_list is None:
                for v in data.values():
                    if isinstance(v, list):
                        prec_list = v
                        break

        if not prec_list:
            print(f"[{year}] page {page} 더 이상 결과 없음 (빈 리스트).")
            break

        print(f"[{year}] page {page} : {len(prec_list)}건")

        # jsonl 형식으로 저장
        for item in prec_list:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 마지막 페이지 chekc
        if len(prec_list) < DISPLAY:
            print(f"[{year}] page {page} 에서 이 연도 마지막 페이지로 판단, 종료.")
            break

        page += 1
        time.sleep(0.3)

def main():
    output_file = "prec_2000_2025.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for year in range(START_YEAR, END_YEAR + 1):
            fetch_year(year, f)

    print(f"모든 판례를 {output_file} 파일에 jsonl 형식으로 저장완료.")


if __name__ == "__main__":
    main()
