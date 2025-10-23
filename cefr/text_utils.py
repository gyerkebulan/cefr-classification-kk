import re

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
CYRILLIC_WORD = re.compile(r"^[а-яёіїңғүұәөһіъь]+(?:['’\-][а-яёіїңғүұәөһіъь]+)*$", re.IGNORECASE)


def tokenize_words(text):
    """Return non-empty tokens captured by the shared word pattern."""
    return [token for token in WORD_PATTERN.findall(text)]


def is_cyrillic_token(token):
    token = token.strip().lower()
    if not token:
        return False
    return bool(CYRILLIC_WORD.fullmatch(token.replace("’", "'")))


__all__ = ["tokenize_words", "is_cyrillic_token"]
