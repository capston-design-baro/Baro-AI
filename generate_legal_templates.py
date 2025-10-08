#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate few-shot style templates and a production-ready compose prompt
for Korean criminal complaint drafts (e.g., 사기죄, 모욕죄).

This script helps you transform ~20 example complaints into:
  1) Style patterns (with placeholders) extracted from your examples
  2) A few-shot prompt file per offense that enforces proper legal style
  3) A helper function snippet you can paste into your codebase

Assumptions about your dataset (choose ONE of the two):
  A) Directory of JSON files (*.json), each with keys:
     {
       "offense": "사기" | "모욕",
       "facts_text": "...범죄사실 줄글...",
       "reason_text": "...고소이유 줄글..."
     }

  B) A CSV file with headers: offense,facts_text,reason_text

Outputs:
  ./out/templates/{offense}_patterns.txt        # deduplicated style patterns with placeholders
  ./out/prompts/{offense}_compose_prompt.txt    # few-shot compose prompt ready to use
  ./out/snippets/{offense}_compose_snippet.py   # drop-in compose() helper (prompt builder)

Usage examples:
  python generate_legal_templates.py --json_dir data/scam_json
  python generate_legal_templates.py --csv data/complaints.csv

No external dependencies; standard library only.
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Basic text utils
def normalize_ws(text: str) -> str:
    text = re.sub(r"\r\n?|\u2028|\u2029", "\n", text)
    text = re.sub(r"\t+", " ", text)
    # collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()

# PII masking / placeholderization

# Very lightweight heuristics; adjust to your data as needed.
RE_DATE = re.compile(r"(20\d{2}|19\d{2})[.\-/년 ]\s?(\d{1,2})[.\-/월 ]\s?(\d{1,2})(?:[.\-/일 ]\s?(?:경|자정|오전|오후|\d{1,2}:\d{2})?)?" )
RE_TIME = re.compile(r"(\d{1,2}:\d{2})경?")
RE_MONEY = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\s*(만원|원|억|조)")
RE_ACCOUNT = re.compile(r"(\d{2,4}-\d{3,4}-\d{3,4}-\d{3,4})")
RE_PHONE = re.compile(r"(01[016789]-?\d{3,4}-?\d{4})")
RE_ADDR = re.compile(r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)[^\s,\.]{0,20}")
RE_COMPANY = re.compile(r"\(주\)[^\s\)]+|[A-Za-z가-힣0-9]{2,}주식회사")
RE_PERSON = re.compile(r"[가-힣]{2,3}씨?|[가-힣]{2,3} (군|양|님)")

PLACEHOLDERS = [
    (RE_DATE, "{DATE}"),
    (RE_TIME, "{TIME}"),
    (RE_MONEY, "{AMOUNT}"),
    (RE_ACCOUNT, "{ACCOUNT}"),
    (RE_PHONE, "{PHONE}"),
    (RE_COMPANY, "{COMPANY}"),
    (RE_PERSON, "{PERSON}"),
]

# Soft masking of addresses; keep city/province only
RE_ADDR_DETAIL = re.compile(r"(\d{1,4}번지|\d{1,4}-\d{1,4}|[가-힣0-9]{1,10}(길|로)\s?\d{1,4})")


def to_bullets(text: str) -> List[str]:
    """Ensure each leading '○ ' bullet line is separated; if none, split by sentences."""
    text = normalize_ws(text)
    lines = [l for l in text.split("\n") if l]
    if any(l.strip().startswith("○") for l in lines):
        bullets = []
        for l in lines:
            if l.strip():
                if l.strip().startswith("○"):
                    bullets.append(l.strip())
                else:
                    # attach to previous bullet
                    if bullets:
                        bullets[-1] += " " + l.strip()
                    else:
                        bullets.append("○ " + l.strip())
        return bullets
    # fallback: sentence split (very rough)
    sents = re.split(r"(?<=[.?!\u3002\uFF0E])\s+|\n", text)
    sents = [s.strip() for s in sents if s.strip()]
    return ["○ " + s for s in sents]


def placeholderize(line: str) -> str:
    line = RE_ADDR_DETAIL.sub("{ADDR_DETAIL}", line)
    for patt, ph in PLACEHOLDERS:
        line = patt.sub(ph, line)
    # normalize multiple spaces
    line = re.sub(r"\s{2,}", " ", line).strip()
    return line


def extract_patterns(facts_text: str, reason_text: str) -> Tuple[List[str], List[str]]:
    facts_bullets = [placeholderize(b) for b in to_bullets(facts_text)]
    reason_bullets = [placeholderize(b) for b in to_bullets(reason_text)]
    # de-duplicate while preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            k = x
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out
    return dedup(facts_bullets), dedup(reason_bullets)

# IO helpers
def load_records_from_json_dir(d: Path) -> List[Dict]:
    records = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if all(k in obj for k in ("offense", "facts_text", "reason_text")):
                records.append(obj)
        except Exception:
            continue
    return records


def load_records_from_csv(f: Path) -> List[Dict]:
    recs = []
    with f.open("r", encoding="utf-8") as rf:
        rdr = csv.DictReader(rf)
        for row in rdr:
            if row.get("offense") and row.get("facts_text") and row.get("reason_text"):
                recs.append({
                    "offense": row["offense"].strip(),
                    "facts_text": row["facts_text"],
                    "reason_text": row["reason_text"],
                })
    return recs

# ------------------------------
# Few-shot prompt builder
PROMPT_HEADER = (
    "당신은 한국 형사 고소장의 초안을 작성하는 전문 비서입니다.\n"
    "아래의 '사건 요소(JSON)'와 '증거 메모'를 참고하여,\n"
    "'범죄사실'과 '고소이유'를 한국어 **줄글(문단 형태)**로 작성하십시오.\n\n"
    "규칙:\n"
    "- 실제 고소장 문체로, 각 문단은 '○' 기호로 시작합니다.\n"
    "- '누가/언제/어디서' 같은 메타 항목명을 노출하지 말고, 문장 속에 자연스럽게 녹여 쓰십시오.\n"
    "- 보고서체/목록체/메모체 금지. 완전한 문장으로, 중립적·객관적 서술을 유지하십시오.\n"
    "- 불명확한 부분은 '□(확인 필요)'로 표기하십시오.\n"
    "- 죄명·조문은 RAG로 제공된 스니펫 내에서만 언급하십시오(환각 금지).\n\n"
)

FEWSHOT_INTRO = (
    "[문체 참고 예시]\n"
    "아래는 실제 출력 형식의 예입니다(데이터는 비식별·가공됨).\n"
)

PROMPT_FOOTER = (
    "\n[출력 형식]\n"
    "=== 범죄사실 ===\n"
    "○ ...\n"
    "○ ...\n\n"
    "=== 고소이유 ===\n"
    "○ ...\n"
    "○ ...\n"
)


def build_fewshot_block(examples: List[Dict], k: int = 2) -> str:
    """Pick k examples per offense and build a few-shot block with masked placeholders."""
    rng = random.Random(42)
    by_offense: Dict[str, List[Dict]] = {}
    for r in examples:
        by_offense.setdefault(r["offense"], []).append(r)

    blocks = []
    for offense, recs in by_offense.items():
        rng.shuffle(recs)
        sub = recs[:k] if len(recs) >= k else recs
        for rec in sub:
            f_pats, r_pats = extract_patterns(rec["facts_text"], rec["reason_text"])
            block = [f"# 예시({offense})"]
            block.append("=== 범죄사실 ===")
            block.extend(f_pats[:4])  # first 4 bullets
            block.append("")
            block.append("=== 고소이유 ===")
            block.extend(r_pats[:3])  # first 3 bullets
            blocks.append("\n".join(block))
    return FEWSHOT_INTRO + "\n\n".join(blocks) + "\n\n"

# Compose snippet generator
COMPOSE_SNIPPET_TMPL = """
# --- Paste this helper into your codebase ---

def build_compose_prompt(offense: str, statute_ref: str, collected: dict, evidence: list[str], fewshot_block: str) -> str:
    header = {header!r}
    footer = {footer!r}
    sys = (
        f"[죄명] {{offense}} ({{statute_ref}})\n"
        f"[사건 요소(JSON)]\n{{collected}}\n\n"
        f"[증거 메모]\n{{evidence}}\n\n"
    )
    return header + fewshot_block + sys + footer
""".replace("{header}", json.dumps(PROMPT_HEADER, ensure_ascii=False))\
   .replace("{footer}", json.dumps(PROMPT_FOOTER, ensure_ascii=False))


# Main pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, help="Directory of JSON records")
    ap.add_argument("--csv", type=str, help="CSV file of records")
    ap.add_argument("--shots", type=int, default=2, help="Few-shot examples per offense")
    args = ap.parse_args()

    if not args.json_dir and not args.csv:
        ap.error("Provide --json_dir or --csv")

    records: List[Dict]
    if args.json_dir:
        records = load_records_from_json_dir(Path(args.json_dir))
    else:
        records = load_records_from_csv(Path(args.csv))

    if not records:
        raise SystemExit("No records loaded. Check your input path.")

    out_root = Path("out")
    (out_root / "templates").mkdir(parents=True, exist_ok=True)
    (out_root / "prompts").mkdir(parents=True, exist_ok=True)
    (out_root / "snippets").mkdir(parents=True, exist_ok=True)

    # group by offense
    by_offense: Dict[str, List[Dict]] = {}
    for r in records:
        by_offense.setdefault(r["offense"], []).append(r)

    # build global few-shot for each offense separately
    for offense, recs in by_offense.items():
        fewshot_block = build_fewshot_block(recs, k=args.shots)
        # Save prompt
        prompt_path = out_root / "prompts" / f"{offense}_compose_prompt.txt"
        prompt_text = PROMPT_HEADER + fewshot_block + PROMPT_FOOTER
        prompt_path.write_text(prompt_text, encoding="utf-8")

        # Save style patterns (aggregate from all records)
        all_f, all_r = [], []
        for rec in recs:
            f_p, r_p = extract_patterns(rec["facts_text"], rec["reason_text"])
            all_f.extend(f_p)
            all_r.extend(r_p)
        # dedup
        def dedup(seq: List[str]) -> List[str]:
            seen = set(); out=[]
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        patterns_path = out_root / "templates" / f"{offense}_patterns.txt"
        patterns_text = "\n".join(["# 범죄사실 패턴"] + dedup(all_f) + ["", "# 고소이유 패턴"] + dedup(all_r))
        patterns_path.write_text(patterns_text, encoding="utf-8")

        # Save compose snippet
        snippet_path = out_root / "snippets" / f"{offense}_compose_snippet.py"
        snippet_path.write_text(COMPOSE_SNIPPET_TMPL, encoding="utf-8")

    print("Done. Check ./out/{templates,prompts,snippets}")

if __name__ == "__main__":
    main()
