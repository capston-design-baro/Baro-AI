
# --- Paste this helper into your codebase ---

def build_compose_prompt(offense: str, statute_ref: str, collected: dict, evidence: list[str], fewshot_block: str) -> str:
    header = {header!r}
    footer = {footer!r}
    sys = (
        f"[죄명] {{offense}} ({{statute_ref}})
"
        f"[사건 요소(JSON)]
{{collected}}

"
        f"[증거 메모]
{{evidence}}

"
    )
    return header + fewshot_block + sys + footer
