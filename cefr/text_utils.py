import re

CYRILLIC_WORD = re.compile(r"^[а-яёіїңғүұәөһіъь]+$", re.IGNORECASE)

def tokenize_words(text):
    """Return non-empty whitespace-separated tokens."""
    return tuple(part for part in text.strip().split() if part)


def is_cyrillic_token(token):
    token = token.strip().lower()
    if not token:
        return False
    normalized = token.replace("-", "").replace("'", "")
    return bool(CYRILLIC_WORD.fullmatch(normalized))


__all__ = ["tokenize_words", "is_cyrillic_token"]
