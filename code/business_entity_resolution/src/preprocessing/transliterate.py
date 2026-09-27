"""Local transliteration (Sec 5.2). Native + Latin, never overwrites native.

Offline only: Latin-diacritic folding + a small generic Devanagari->Latin map
learned from training-data patterns (no external DB/API). Everything else
passes through unchanged (French accents folded, CJK preserved).
"""
import unicodedata

_DEV_MAP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
            "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ं": "n", "ः": "h",
            "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "c", "छ": "ch",
            "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
            "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
            "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y",
            "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
            "ह": "h", "्": "", "ा": "a", "ि": "i", "ी": "ii", "ु": "u",
            "ू": "uu", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri"}


def transliterate_local(s):
    """Auxiliary Latin view; empty-safe; never used to delete native text."""
    if not s:
        return ""
    s = str(s)
    try:
        from .normalize import HAS_UNIDECODE, _unidecode
        if HAS_UNIDECODE:
            out = _unidecode(s)
            if out and out.strip():
                return " ".join(out.lower().split())
    except Exception:
        pass
    buf = []
    for ch in s:
        if ch in _DEV_MAP:
            buf.append(_DEV_MAP[ch])
        elif ord(ch) < 384:
            n = unicodedata.normalize("NFKD", ch)
            buf.append("".join(c for c in n if not unicodedata.combining(c)))
        else:
            buf.append(ch)
    return " ".join("".join(buf).lower().split())
