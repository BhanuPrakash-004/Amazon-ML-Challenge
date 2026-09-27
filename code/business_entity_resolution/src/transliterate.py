"""LOCAL transliteration helper (offline, no APIs, no external data).

Original text is NEVER replaced: transliteration is an auxiliary
representation used for retrieval/features alongside the original.

Strategy (stdlib only, deterministic):
- Map Devanagari block (U+0900-U+097F) via a compact hand-built table
  covering common letters + vowel signs + anusvara/visarga.
- Fold Latin diacritics (é->e) for the auxiliary form only.
- Unknown chars pass through unchanged; failure never drops the original.
"""
import unicodedata

_DEV = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "am", "अः": "ah",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "ळ": "l", "क्ष": "ksh", "ज्ञ": "gya",
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "m", "ः": "h",
    "ँ": "n", "्": "", "ॉ": "o", "़": "", "।": " ", "॥": " ",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def transliterate_local(s):
    """Return auxiliary Latin form; never raises; never returns empty for non-empty input."""
    if s is None:
        return ""
    s = str(s)
    if not s:
        return ""
    try:
        out = []
        for ch in s:
            if ch in _DEV:
                out.append(_DEV[ch])
            elif ord(ch) < 384:
                n = unicodedata.normalize("NFKD", ch)
                out.append("".join(c for c in n if not unicodedata.combining(c)))
            else:
                out.append(ch)  # preserve unknown scripts (French handled by fold above)
        t = "".join(out)
        t = " ".join(t.split())
        return t if t else s
    except Exception:
        return s


def has_non_latin(s):
    try:
        return any(ord(c) > 127 for c in str(s))
    except Exception:
        return False
