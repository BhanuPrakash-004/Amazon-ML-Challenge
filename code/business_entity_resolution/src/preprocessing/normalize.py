"""Source-aware normalization (Sec 5). Multi-representation, never replaces raw.

Per record:
  country_norm, name_raw/norm/translit/core/compact/tokens,
  address_raw/norm/translit/compact/tokens, house_number, numeric_tokens,
  numeric_signature, postal_candidate, missingness flags.
Unicode NFKC + lowercase + whitespace/punct norm. Legal-suffix reduction only
in name_core (terminal strip). Country-aware address rules are generic
(rd/st/ave/h.no/unit) — no external DB (Sec 5.4).
"""
import re
import unicodedata

LEGAL_SUFFIXES = {"pvt", "private", "ltd", "limited", "llc", "llp", "inc",
                  "incorporated", "corp", "corporation", "co", "company", "plc"}

NAME_PUNCT_MAP = {"&": " and "}
ADDR_ABBR = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
             "blvd": "boulevard", "ln": "lane", "dr": "drive", "apt": "apartment",
             "ste": "suite", "fl": "floor", "bldg": "building", "opp": "opposite",
             "nr": "near", "marg": "road", "nagar": "nagar", "h": "house"}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

try:
    from unidecode import unidecode as _unidecode
    HAS_UNIDECODE = True
except Exception:
    HAS_UNIDECODE = False
    _unidecode = None


def unicode_norm(s):
    s = "" if s is None else str(s)
    return unicodedata.normalize("NFKC", s)


def lower_ws(s):
    return _WS.sub(" ", s.lower()).strip()


def fold_punct(s):
    for k, v in NAME_PUNCT_MAP.items():
        s = s.replace(k, v)
    s = _PUNCT.sub(" ", s).replace("_", " ")
    return _WS.sub(" ", s).strip()


def normalize_country(s):
    c = lower_ws(fold_punct(unicode_norm(s)))
    alias = {"usa": "united states", "u s a": "united states",
             "united states of america": "united states", "us": "united states",
             "u s": "united states", "america": "united states",
             "bharat": "india", "hindustan": "india", "in": "india",
             "fr": "france", "fra": "france", "french republic": "france",
             "republique francaise": "france"}
    return alias.get(c, c)


def normalize_name(s):
    return lower_ws(fold_punct(unicode_norm(s)))


def normalize_address(s):
    t = lower_ws(fold_punct(unicode_norm(s)))
    toks = [ADDR_ABBR.get(w, w) for w in t.split()]
    # h.no / hno / no forms -> house
    out = []
    for w in toks:
        if w in ("hno", "hno.", "h", "no", "number", "#"):
            out.append("house")
        else:
            out.append(w)
    return " ".join(out)


def name_core(name_norm):
    toks = name_norm.split()
    for _ in range(3):
        if toks and toks[-1] in LEGAL_SUFFIXES:
            toks.pop()
        else:
            break
    return " ".join(toks)


def name_compact(name_norm):
    return re.sub(r"\s+", "", name_norm)


def address_compact(addr_norm):
    return re.sub(r"\s+", "", addr_norm)


def tokenize(s):
    return s.split() if s else []


def name_toks(rec):
    """Name tokens; splits name_norm on the fly for lean records (Sec 33.2)."""
    t = rec.get("name_tokens")
    if t:
        return t
    return rec.get("name_norm", "").split()


def addr_toks(rec):
    t = rec.get("address_tokens")
    if t:
        return t
    return rec.get("address_norm", "").split()


def normalize_record(name, addr, country="", lean=False):
    """Full (default) or lean (targets at scale) representation.

    lean=True drops raw strings and pre-split token lists (re-derivable via
    name_toks/addr_toks) to cut ~50% memory on 10M-row stores (Sec 33.2).
    All blocking/feature/embedding fields are preserved.
    """
    from .transliterate import transliterate_local
    from .numbers import extract_all as _num
    name_raw = "" if name is None else str(name)
    addr_raw = "" if addr is None else str(addr)
    nn = normalize_name(name_raw)
    aa = normalize_address(addr_raw)
    cc = normalize_country(country)
    num = _num(name_raw + " " + addr_raw)
    rec = {
        "country_norm": cc,
        "name_norm": nn,
        "name_translit": transliterate_local(nn),
        "name_core": name_core(nn), "name_compact": name_compact(nn),
        "address_norm": aa,
        "address_translit": transliterate_local(aa),
        "address_compact": address_compact(aa),
        "house_number": num["house_number"],
        "numeric_tokens": num["numeric_tokens"],
        "numeric_signature": num["numeric_signature"],
        "postal_candidate": num["postal_candidate"],
        "address_missing": 1 if not addr_raw.strip() else 0,
    }
    if lean:
        return rec
    rec.update({
        "name_raw": name_raw,
        "name_tokens": tokenize(nn),
        "name_rare_tokens": [],  # populated post-hoc by rarity.attach_rare
        "address_raw": addr_raw,
        "address_tokens": tokenize(aa),
        "address_rare_tokens": [],  # populated post-hoc by rarity.attach_rare
    })
    return rec
