"""Numeric extraction (Sec 6, high priority).

Extracts house_number, all numeric tokens, sorted signature, postal candidates
(conservative: 5-6 digit only + context), fractions (14/2), alphanumeric
house ids (AF-684, AF-0684 -> af684 canonical).
"""
import re

_DIG = re.compile(r"\d+")
_HOUSE_PAT = re.compile(r"(?i)\b(?:h\.?\s*no\.?|house\s*no\.?|no\.?|#|plot|shop|unit|flat|apt\.?)?\s*"
                        r"(\d{1,6}[A-Za-z]?(?:\s*[/\-]\s*\d{1,6}[A-Za-z]?)?)")
_FRAC = re.compile(r"(\d+)\s*/\s*(\d+)")
_ALNUM_HOUSE = re.compile(r"(?i)\b([A-Za-z]{1,4})[\-\s]?0*(\d{1,6}[A-Za-z]?)\b")
_PIN6 = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_ZIP5 = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")


def canon_house(s):
    s = re.sub(r"[\s\-/]+", "", str(s).lower())
    s = re.sub(r"^0+(?=\d)", "", s)
    m = _ALNUM_HOUSE.match(str(s))
    return s


def extract_all(text):
    text = "" if text is None else str(text)
    nums = _DIG.findall(text)
    # house: first strong house-like match
    house = ""
    m = _HOUSE_PAT.search(text)
    if m:
        house = canon_house(m.group(1))
    # alnum house ids
    for a, d in _ALNUM_HOUSE.findall(text):
        if len(d.strip("0")) >= 1:
            house = house or canon_house(a + d)
            break
    fracs = _FRAC.findall(text)
    pins = _PIN6.findall(text)
    zips = _ZIP5.findall(text)
    postal = sorted(set(pins) | set(zips))
    sig = tuple(sorted(nums))
    return {"house_number": house, "numeric_tokens": nums,
            "numeric_signature": "|".join(sig),
            "postal_candidate": postal[0] if postal else "",
            "postal_all": postal, "fractions": fracs}
