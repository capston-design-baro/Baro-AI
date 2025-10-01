# services/openai_client.py
from typing import Optional
from openai import OpenAI
from cfg import settings

_client: Optional[OpenAI] = None

def client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요.")
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client

def _extract_output_text(resp) -> str:
    """
    Responses API와 Chat Completions API를 모두 커버하는 출력 파서.
    """
    # Responses API (신규): resp.output_text 속성 제공
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Responses API (수동 파싱)
    try:
        parts = resp.output[0].content
        if parts and hasattr(parts[0], "text"):
            return parts[0].text.value
    except Exception:
        pass

    # Chat Completions
    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    return ""

def respond(model: str, system: str, user: str) -> str:
    c = client()

    try:
        resp = c.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _extract_output_text(resp).strip()
    except TypeError:
        pass
    except Exception:
        pass
