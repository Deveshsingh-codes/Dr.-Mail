import re
_TAGS = re.compile(r"<(script|style).*?</\1>|<[^>]+>", re.S | re.I)
_WS = re.compile(r"\s+")

def clean_text(text) -> str:
    """Strip HTML and normalise whitespace. URLs, numbers and symbols are kept on purpose."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", _TAGS.sub(" ", text)).strip()
