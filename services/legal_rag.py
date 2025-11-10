from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests import Response
from xml.etree import ElementTree as ET

from cfg import settings
from services.openai_client import respond


@dataclass
class RetrievedDoc:
    kind: str       
    id: str
    title: str
    snippet: str
    url: str


BASE = Path(__file__).resolve().parents[1]
OFFENSE_DIR = BASE / "data" / "offenses"
DEFAULT_LAW_BASE = "https://www.law.go.kr/DRF"


def list_offense_defs() -> List[Dict[str, str]]:
    """List available offenses from YAML with minimal fields."""

    out: List[Dict[str, str]] = []
    for p in sorted(OFFENSE_DIR.glob("*.yaml")):
        try:
            import yaml  # local import to avoid unused dep when not needed

            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            key = str(doc.get("offense") or p.stem)
            title = str(doc.get("title_ko") or key)
            statute_ref = str(doc.get("statute_ref") or "")
            out.append({"key": key, "title_ko": title, "statute_ref": statute_ref})
        except Exception:
            continue
    return out


class LawRetriever:
    #Lightweight client for https://www.law.go.kr/DRF (국가법령정보센터) Open API.
    
    STATUTE_TITLE_KEYS = ["법령명한글", "법령명", "법령한글명", "lawname", "title"]
    STATUTE_SNIPPET_KEYS = ["조문내용", "조문", "내용", "content", "제목"]
    STATUTE_URL_KEYS = ["법령링크", "법령url", "lawurl", "url", "link"]
    STATUTE_ID_KEYS = ["법령id", "법령일련번호", "lawid", "법령번호"]

    CASE_TITLE_KEYS = ["판례명", "사건명", "precname", "title"]
    CASE_SNIPPET_KEYS = ["판시사항", "판결요지", "요지", "요약", "content"]
    CASE_ID_KEYS = ["사건번호", "판례일련번호", "precid", "precedentseq", "id"]
    CASE_URL_KEYS = ["판례링크", "판례본문url", "판례url", "url", "link"]

    def __init__(
        self,
        base: Optional[str] = None,
        key: Optional[str] = None,
        timeout: float = 6.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base = (base or settings.LAW_API_BASE or DEFAULT_LAW_BASE).rstrip("/")
        self.key = key or settings.LAW_API_KEY or ""
        self.timeout = timeout
        self.session = session or requests.Session()

    # ----- public API -----
    def search_cases(self, query: str, limit: int = 3) -> List[RetrievedDoc]:
        #검색 질의에 해당하는 판례 조회

        if not self._enabled():
            return []
        try:
            data = self._request(
                "precSearch.do",
                {
                    "OC": self.key,
                    "target": "prec",
                    "type": "JSON",
                    "query": query,
                    "display": max(1, min(20, limit * 2)),  # 필터링 여유분 확보
                    "page": 1,
                },
            )
        except Exception:
            return []
        return self._parse_cases(data, limit)

    def search_statutes(self, query: str, limit: int = 3) -> List[RetrievedDoc]:
        #검색 질의에 해당하는 법령 조회

        if not self._enabled():
            return []
        try:
            data = self._request(
                "lawSearch.do",
                {
                    "OC": self.key,
                    "target": "law",
                    "type": "JSON",
                    "query": query,
                    "display": max(1, min(20, limit * 2)),
                    "page": 1,
                },
            )
        except Exception:
            return []
        return self._parse_statutes(data, limit)

    # ----- internals -----
    def _enabled(self) -> bool:
        return bool(self.base and self.key)

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base}/{endpoint.lstrip('/')}"
        resp: Response = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            pass
        return self._xml_to_dict(resp.text)

    def _xml_to_dict(self, xml_text: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return {}

        def _convert(node: ET.Element):
            children = list(node)
            if not children:
                return (node.text or "").strip()
            data: Dict[str, Any] = {}
            for child in children:
                value = _convert(child)
                if child.tag in data:
                    current = data[child.tag]
                    if not isinstance(current, list):
                        data[child.tag] = [current]
                    data[child.tag].append(value)
                else:
                    data[child.tag] = value
            return data

        return {root.tag: _convert(root)}

    @staticmethod
    def _iter_dict_lists(node: Any) -> Iterable[List[Dict[str, Any]]]:
        queue: List[Any] = [node]
        while queue:
            cur = queue.pop(0)
            if isinstance(cur, list):
                if cur and all(isinstance(x, dict) for x in cur):
                    yield cur
                queue.extend(cur)
            elif isinstance(cur, dict):
                queue.extend(cur.values())

    def _flatten(self, entry: Any) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        stack: List[Any] = [entry]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                    else:
                        flat.setdefault(key, value)
            elif isinstance(node, list):
                for value in node:
                    if isinstance(value, (dict, list)):
                        stack.append(value)
        return flat

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        return ""

    @staticmethod
    def _normalize_key(key: str) -> str:
        return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(key or "")).lower()

    def _pick_value(self, entry: Dict[str, Any], key_candidates: List[str]) -> str:
        targets = [self._normalize_key(k) for k in key_candidates]
        for candidate in targets:
            if not candidate:
                continue
            for actual_key, actual_val in entry.items():
                norm_actual = self._normalize_key(actual_key)
                if not norm_actual:
                    continue
                if norm_actual == candidate or candidate in norm_actual:
                    val = self._to_text(actual_val)
                    if val:
                        return val
        return ""

    def _parse_statutes(self, payload: Any, limit: int) -> List[RetrievedDoc]:
        docs: List[RetrievedDoc] = []
        for block in self._iter_dict_lists(payload):
            for entry in block:
                flat = self._flatten(entry)
                title = self._pick_value(flat, self.STATUTE_TITLE_KEYS)
                snippet = self._pick_value(flat, self.STATUTE_SNIPPET_KEYS)
                if not title and not snippet:
                    continue
                doc_id = self._pick_value(flat, self.STATUTE_ID_KEYS) or title or "statute"
                url = self._pick_value(flat, self.STATUTE_URL_KEYS)
                docs.append(
                    RetrievedDoc(
                        kind="statute",
                        id=doc_id,
                        title=title or doc_id,
                        snippet=snippet or "",
                        url=url or "",
                    )
                )
                if len(docs) >= limit:
                    return docs
        return docs

    def _parse_cases(self, payload: Any, limit: int) -> List[RetrievedDoc]:
        docs: List[RetrievedDoc] = []
        for block in self._iter_dict_lists(payload):
            for entry in block:
                flat = self._flatten(entry)
                title = self._pick_value(flat, self.CASE_TITLE_KEYS)
                snippet = self._pick_value(flat, self.CASE_SNIPPET_KEYS)
                if not title and not snippet:
                    continue
                doc_id = self._pick_value(flat, self.CASE_ID_KEYS) or title or "case"
                url = self._pick_value(flat, self.CASE_URL_KEYS)
                docs.append(
                    RetrievedDoc(
                        kind="case",
                        id=doc_id,
                        title=title or doc_id,
                        snippet=snippet or "",
                        url=url or "",
                    )
                )
                if len(docs) >= limit:
                    return docs
        return docs


# Classification with RAG context
CLASSIFY_SYS = (
    "You are a Korean legal assistant that classifies an input narrative "
    "into one of the supported offenses. Use only the provided context; "
    "if uncertain, choose the closest match and lower confidence."
)


def _build_context_block(text: str, statutes: List[RetrievedDoc], cases: List[RetrievedDoc]) -> str:
    lines = ["[사용자 서술]", text.strip(), ""]
    if statutes:
        lines.append("[관련 법령 요약]")
        for d in statutes:
            lines.append(f"- {d.title}: {d.snippet} ({d.url})")
        lines.append("")
    if cases:
        lines.append("[관련 판례 요약]")
        for d in cases:
            lines.append(f"- {d.title}: {d.snippet} ({d.url})")
        lines.append("")
    return "\n".join(lines)


def _build_offense_choices(offenses: List[Dict[str, str]]) -> str:
    lines = ["[선택 가능한 죄명 목록]"]
    for o in offenses:
        ref = f" ({o['statute_ref']})" if o.get("statute_ref") else ""
        lines.append(f"- {o['key']}: {o['title_ko']}{ref}")
    lines.append("")
    lines.append(
        "[출력 형식]\n"
        "{\n  \"prediction\": \"<offense_key>\",\n  \"candidates\": [\n"
        "    {\"offense\": \"<offense_key>\", \"confidence\": 0.0, \"rationale\": \"...\"}\n"
        "  ]\n}"
    )
    return "\n".join(lines)


def _fallback_keyword_classifier(text: str, offenses: List[Dict[str, str]]) -> Tuple[str, List[Tuple[str, float, str]]]:
    """Very rough keyword-based classifier for offline fallback."""

    t = (text or "").lower()
    scores: Dict[str, float] = {o["key"]: 0.0 for o in offenses}

    fraud_kw = ["사기", "기망", "송금", "이체", "입금", "보이스피싱", "거래", "계좌", "돈", "수금"]
    insult_kw = ["모욕", "욕", "비하", "욕설", "모욕죄", "비방"]

    for w in fraud_kw:
        if w in t:
            scores["fraud"] = scores.get("fraud", 0.0) + 0.2
    for w in insult_kw:
        if w in t:
            scores["insult"] = scores.get("insult", 0.0) + 0.25

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked[0][0] if ranked else (offenses[0]["key"] if offenses else "")
    cands = [(k, s, f"keyword matches for {k}") for k, s in ranked if s > 0]
    if not cands:
        cands = [(top, 0.3, "no strong keywords; defaulting to closest match")]
    return top, cands[:3]


def classify_offense_with_rag(
    user_text: str,
    retriever: Optional[LawRetriever] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Classify offense key using retrieved statutes + cases as context."""

    offenses = list_offense_defs()
    if not offenses:
        return {"prediction": "", "candidates": [], "retrieval": {"statutes": [], "cases": []}}

    retriever = retriever or LawRetriever()

    statutes = retriever.search_statutes(user_text, limit=top_k) if retriever else []
    cases = retriever.search_cases(user_text, limit=top_k) if retriever else []

    context = _build_context_block(user_text, statutes, cases)
    choices = _build_offense_choices(offenses)

    prompt = (
        "아래 사용자 서술과 검색된 법령/판례 요약을 참고하여, 제공된 죄명 목록 중 가장 적합한 죄명을 선택하십시오.\n"
        "근거를 간단히 한국어로 설명하고, 신뢰도(confidence, 0.0~1.0)를 추정하십시오.\n\n"
        f"{choices}\n\n"
        f"{context}\n"
    )

    out = respond(settings.OPENAI_CHAT_MODEL, CLASSIFY_SYS, prompt)
    try:
        obj = json.loads(out)
        prediction = str(obj.get("prediction") or "").strip()
        cands = obj.get("candidates") or []
        norm_cands = []
        for c in cands:
            try:
                norm_cands.append(
                    {
                        "offense": str(c.get("offense") or "").strip(),
                        "confidence": float(c.get("confidence") or 0.0),
                        "rationale": str(c.get("rationale") or "").strip(),
                    }
                )
            except Exception:
                continue
        if not prediction and norm_cands:
            prediction = norm_cands[0]["offense"]
    except Exception:
        pred, kw_cands = _fallback_keyword_classifier(user_text, offenses)
        norm_cands = [
            {"offense": k, "confidence": min(1.0, max(0.0, s)), "rationale": r}
            for k, s, r in kw_cands
        ]
        prediction = pred

    return {
        "prediction": prediction,
        "candidates": norm_cands[:3],
        "retrieval": {
            "statutes": [d.__dict__ for d in statutes],
            "cases": [d.__dict__ for d in cases],
        },
    }

