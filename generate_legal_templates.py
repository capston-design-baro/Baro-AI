

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ====== 경로 설정 ======
BASE = Path(__file__).resolve().parent        # <= 여기!
PLAINT_DIR = BASE / "data" / "plaint"
FULLTEXT_XLSX = BASE / "data" / "fulltext.xlsx"
OUT_ROOT = BASE / "out"                     # 결과물 저장 폴더


def normalize_ws(text: str) -> str:
    text = re.sub(r"\r\n?|\u2028|\u2029", "\n", text)
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


RE_DATE = re.compile(
    r"(20\d{2}|19\d{2})[.\-/년 ]\s?(\d{1,2})[.\-/월 ]\s?(\d{1,2})"
    r"(?:[.\-/일 ]\s?(?:경|자정|오전|오후|\d{1,2}:\d{2})?)?"
)
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

RE_ADDR_DETAIL = re.compile(r"(\d{1,4}번지|\d{1,4}-\d{1,4}|[가-힣0-9]{1,10}(길|로)\s?\d{1,4})")


def to_bullets(text: str) -> List[str]:
    text = normalize_ws(text)
    lines = [l for l in text.split("\n") if l]
    if any(l.strip().startswith("○") for l in lines):
        bullets = []
        for l in lines:
            if l.strip():
                if l.strip().startswith("○"):
                    bullets.append(l.strip())
                else:
                    if bullets:
                        bullets[-1] += " " + l.strip()
                    else:
                        bullets.append("○ " + l.strip())
        return bullets

    sents = re.split(r"(?<=[.?!\u3002\uFF0E])\s+|\n", text)
    sents = [s.strip() for s in sents if s.strip()]
    return ["○ " + s for s in sents]


def placeholderize(line: str) -> str:
    line = RE_ADDR_DETAIL.sub("{ADDR_DETAIL}", line)
    for patt, ph in PLACEHOLDERS:
        line = patt.sub(ph, line)
    line = re.sub(r"\s{2,}", " ", line).strip()
    return line


def dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def map_offense_code(name: str) -> str:
    name = (name or "").strip()
    if name in ("사기", "사기죄"):
        return "fraud"
    if name in ("모욕", "모욕죄"):
        return "insult"
    return name  # 이미 fraud/insult 등으로 되어 있으면 그대로


def load_records_from_json_dir(d: Path) -> List[Dict]:
    """fraud1.json, fraud2.json 등의 샘플을 읽어옴."""
    records: List[Dict] = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if all(k in obj for k in ("offense", "facts_text", "reason_text")):
            records.append(
                {
                    "offense": map_offense_code(obj["offense"]),
                    "facts_text": obj["facts_text"],
                    "reason_text": obj["reason_text"],
                }
            )
    return records


def load_fulltext_xlsx(path: Path, offense_code: str = "fraud") -> List[Dict]:
    if not path.exists():
        return []

    df = pd.read_excel(path)
    col = "complaint_text"
    if col not in df.columns:
        raise ValueError(f"엑셀에 '{col}' 컬럼이 없습니다.")

    recs: List[Dict] = []
    for _, row in df.iterrows():
        txt = str(row[col]) if row[col] is not None else ""
        txt = txt.strip()
        if not txt:
            continue
        recs.append({"offense": offense_code, "full_text": txt})
    return recs


def extract_patterns(facts_text: str, reason_text: str) -> Tuple[List[str], List[str]]:
    facts_bullets = [placeholderize(b) for b in to_bullets(facts_text)]
    reason_bullets = [placeholderize(b) for b in to_bullets(reason_text)]
    return dedup(facts_bullets), dedup(reason_bullets)


def extract_style_patterns(full_text: str) -> List[str]:
    bullets = to_bullets(full_text)
    pats = [placeholderize(b) for b in bullets]
    return dedup(pats)


PROMPT_HEADER = (
    "당신은 한국 형사 고소장의 초안을 작성하는 전문 비서입니다.\n"
    "아래의 '사건 요소(JSON)'와 '증거 메모'를 참고하여,\n"
    "'범죄사실'과 '고소이유'를 한국어 줄글(문단 형태)로 작성하십시오.\n"
    "'범죄사실'은 형법 등 처벌법규에 해당하는 사실에 대하여 일시, 장소, 범행방법, "
    "결과 등을 특정하여 필요한 사항만 간략하게 기재해야 하며, "
    "고소인이 알고 있는 지식과 경험, 증거에 의해 사실로 인정되는 내용을 기재하여야 합니다.\n"
    "'범죄사실'의 맨 처음에 '고소인과 피고소인의 관계(어떻게 알게 된 사이인지)에 대해 언급하십시오.\n"
    "'고소이유'에는 피고소인의 범행 경위 및 정황, 고소를 하게 된 동기와 사유 등 "
    "범죄사실을 뒷받침하는 내용을 인과관계에 맞게 시간순으로 구체적으로 기재해야 합니다.\n\n"
    "규칙:\n"
    "- 실제 고소장 문체로, '누가/언제/어디서' 같은 메타 항목명을 노출하지 말고, 문장 속에 자연스럽게 녹여 쓰십시오.\n"
    "- 보고서체/목록체/메모체 금지. 완전한 문장으로, 중립적·객관적 서술을 유지하십시오.\n"
    "- 죄명·조문은 RAG로 제공된 스니펫 내에서만 언급하십시오(환각 금지).\n"
    "- 문체는 형사 고소장 기준으로, 주체는 '고소인', 상대방은 '피고소인'으로 표기합니다.\n"
    "- '원고/피고'라는 단어는 절대 사용하지 않습니다.\n"
)

FEWSHOT_INTRO = (
    "[문체 참고 예시(요약형 샘플)]\n"
    "아래는 '범죄사실/고소이유'가 구분된 예시입니다(데이터는 비식별·가공됨).\n"
)

FULLTEXT_INTRO = (
    "\n[문체 참고 예시(통합 고소장 샘플)]\n"
    "아래는 실제 고소장 전체 단락에서 추출한 문장 패턴입니다. 문체와 서술 흐름을 참고만 하십시오.\n"
)

PROMPT_FOOTER = (
    "\n[출력 형식]\n"
    "4. 범죄사실 \n"
    "○ ...\n"
    "○ ...\n\n"
    "5. 고소이유 \n"
    "○ ...\n"
    "○ ...\n"
)

COMPOSE_SNIPPET_TMPL = """
def build_compose_prompt(offense: str, statute_ref: str, collected: dict, evidence: list[str], fewshot_block: str) -> str:
    header = {header!r}
    footer = {footer!r}
    sys = (
        f"[사건 요소(JSON)]\\n{{collected}}\\n\\n"
        f"[증거 메모]\\n{{evidence}}\\n\\n"
    )
    return header + fewshot_block + sys + footer
""".replace("{header}", json.dumps(PROMPT_HEADER, ensure_ascii=False))\
   .replace("{footer}", json.dumps(PROMPT_FOOTER, ensure_ascii=False))


def build_fewshot_block(
    examples: List[Dict],
    full_style: List[str],
    shots_per_offense: int = 2,
    max_fulltext: int = 15,
) -> str:

    rng = random.Random(42)
    by_offense: Dict[str, List[Dict]] = {}
    for r in examples:
        by_offense.setdefault(r["offense"], []).append(r)

    blocks: List[str] = [FEWSHOT_INTRO]

    for offense, recs in by_offense.items():
        rng.shuffle(recs)
        sub = recs[:shots_per_offense] if len(recs) >= shots_per_offense else recs
        for rec in sub:
            f_pats, r_pats = extract_patterns(rec["facts_text"], rec["reason_text"])
            block = [f"# 예시({offense})", "=== 범죄사실 ==="]
            block.extend(f_pats[:4])
            block.append("")
            block.append("=== 고소이유 ===")
            block.extend(r_pats[:3])
            blocks.append("\n".join(block))

    if full_style:
        blocks.append(FULLTEXT_INTRO)
        for line in full_style[:max_fulltext]:
            blocks.append("○ " + line)

    return "\n\n".join(blocks) + "\n\n"


def main():
    plaint_recs = load_records_from_json_dir(PLAINT_DIR)
    if not plaint_recs:
        raise SystemExit(f"plaint JSON이 없습니다: {PLAINT_DIR}")

    fulltext_recs = load_fulltext_xlsx(FULLTEXT_XLSX, offense_code="fraud")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "templates").mkdir(exist_ok=True)
    (OUT_ROOT / "prompts").mkdir(exist_ok=True)
    (OUT_ROOT / "snippets").mkdir(exist_ok=True)

    by_offense: Dict[str, List[Dict]] = {}
    for r in plaint_recs:
        by_offense.setdefault(r["offense"], []).append(r)

    full_style_by_offense: Dict[str, List[str]] = {}
    if fulltext_recs:
        style_lines: List[str] = []
        for rec in fulltext_recs:
            style_lines.extend(extract_style_patterns(rec["full_text"]))
        full_style_by_offense["fraud"] = dedup(style_lines)

    for offense, recs in by_offense.items():
        style_lines = full_style_by_offense.get(offense, [])
        fewshot_block = build_fewshot_block(
            recs, style_lines, shots_per_offense=2, max_fulltext=15
        )

        prompt_path = OUT_ROOT / "prompts" / f"{offense}_compose_prompt.txt"
        prompt_text = PROMPT_HEADER + fewshot_block + PROMPT_FOOTER
        prompt_path.write_text(prompt_text, encoding="utf-8")

        all_f: List[str] = []
        all_r: List[str] = []
        for rec in recs:
            f_p, r_p = extract_patterns(rec["facts_text"], rec["reason_text"])
            all_f.extend(f_p)
            all_r.extend(r_p)
        all_f = dedup(all_f)
        all_r = dedup(all_r)
        style_only = dedup(style_lines)

        patterns_path = OUT_ROOT / "templates" / f"{offense}_patterns.txt"
        patterns_text = "\n".join(
            ["# 범죄사실 패턴"] + all_f
            + ["", "# 고소이유 패턴"] + all_r
            + ["", "# 통합 문체 패턴"] + style_only
        )
        patterns_path.write_text(patterns_text, encoding="utf-8")

        snippet_path = OUT_ROOT / "snippets" / f"{offense}_compose_snippet.py"
        snippet_path.write_text(COMPOSE_SNIPPET_TMPL, encoding="utf-8")


if __name__ == "__main__":
    main()
