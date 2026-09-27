"""Text normalization for business names / addresses / countries.

V2 API preserved (normalize_name/address/country, core_name, token_set,
extract_numbers, extract_pin_zip, combined_text, basic_clean, ascii_fold).
V3 adds multi-representation records: original + norm + compact + tokens +
transliterated + numeric/postal candidates. Unicode-safe: Devanagari and
French accents preserved; transliteration is auxiliary only.
"""
import re
import unicodedata

# Business suffix / abbreviation expansions (applied on lowercased text)
NAME_ABBR = {
    "corp": "corporation",
    "corpn": "corporation",
    "inc": "incorporated",
    "incorp": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "pvtltd": "private limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "co": "company",
    "corp.": "corporation",
    " Bros ": " brothers ",
    "mfg": "manufacturing",
    "mgf": "manufacturing",
    "ent": "enterprises",
    "enterprizes": "enterprises",
    "intl": "international",
    "intl.": "international",
    "assoc": "associates",
    "assn": "association",
    "dept": "department",
    "govt": "government",
    "tech": "technologies",
    "technol": "technologies",
}

ADDRESS_ABBR = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "fl": "floor",
    "bldg": "building",
    "ste": "suite",
    "apt": "apartment",
    "opp": "opposite",
    "nr": "near",
    "dist": "district",
    "tal": "taluk",
    "tq": "taluk",
    "marg": "road",
}

# Legal suffixes stripped for "core name" comparison (terminal positions only
# enforced in core_name_compact; plain core_name keeps V2 behavior).
LEGAL_SUFFIXES = {
    "corporation", "incorporated", "limited", "private", "company",
    "llc", "llp", "inc", "corp", "ltd", "pvt", "co", "enterprises",
    "enterprise", "industries", "industry", "services", "solutions",
    "systems", "technologies", "technology", "associates", "partners",
    "holdings", "group", "traders", "trading",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")
_PIN6_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")   # India PIN
_ZIP5_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")  # US ZIP
_HOUSENO_RE = re.compile(r"(?<!\w)(\d{1,6}[a-zA-Z]?(?:[/-]\d{1,6}[a-zA-Z]?)?)(?!\w)")


def ascii_fold(s: str) -> str:
    """Fold Latin diacritics (é->e) but preserve non-Latin scripts."""
    if s is None:
        return ""
    s = str(s)
    out = []
    for ch in s:
        if ord(ch) < 384:
            n = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in n if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


def basic_clean(s: str) -> str:
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s).replace("_", " ")
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def expand_abbr_tokens(tokens, mapping):
    out = []
    for t in tokens:
        if t in mapping:
            out.extend(mapping[t].split())
        else:
            out.append(t)
    return out


def normalize_name(s: str) -> str:
    s = basic_clean(s)
    toks = s.split()
    toks = expand_abbr_tokens(toks, NAME_ABBR)
    return " ".join(toks)


def normalize_address(s: str) -> str:
    s = basic_clean(s)
    toks = s.split()
    toks = expand_abbr_tokens(toks, ADDRESS_ABBR)
    return " ".join(toks)


def normalize_country(s: str) -> str:
    if s is None:
        return ""
    c = ascii_fold(str(s)).lower().strip()
    c = _MULTI_SPACE.sub(" ", _PUNCT_RE.sub(" ", c)).strip()
    alias = {
        "usa": "united states", "u s a": "united states",
        "united states of america": "united states", "us": "united states",
        "u s": "united states", "america": "united states",
        "bharat": "india", "hindustan": "india", "in": "india",
        "fr": "france", "fra": "france", "french republic": "france",
        "republique francaise": "france",
    }
    return alias.get(c, c)


def core_name(norm_name: str) -> str:
    toks = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks)


def core_name_compact(norm_name: str) -> str:
    """Conservative: strip legal suffixes from terminal positions only (max 3)."""
    toks = norm_name.split()
    for _ in range(3):
        if toks and toks[-1] in LEGAL_SUFFIXES:
            toks.pop()
        else:
            break
    return " ".join(toks)


def token_set(s: str) -> set:
    return set(s.split()) if s else set()


def extract_numbers(s: str):
    return _DIGITS_RE.findall(str(s) if s is not None else "")


def extract_pin_zip(s: str):
    s = str(s) if s is not None else ""
    pins = _PIN6_RE.findall(s)
    zips = _ZIP5_RE.findall(s)
    return pins, zips


def extract_postal_candidates(s: str):
    """Postal/PIN-like values (5-6 digit) + alphanumeric chunks, local only."""
    s = str(s) if s is not None else ""
    out = set(_PIN6_RE.findall(s)) | set(_ZIP5_RE.findall(s))
    return sorted(out)


def extract_numeric_signature(s: str):
    """Sorted digit-sequence signature for numeric blocking (order-free)."""
    return tuple(sorted(extract_numbers(s)))


def extract_house_numbers(s: str):
    return _HOUSENO_RE.findall(str(s) if s is not None else "")[:8]


def combined_text(name_norm: str, addr_norm: str) -> str:
    return (name_norm + " " + addr_norm).strip()


# ---------------- V3 multi-representation record ----------------

def normalize_record(name, addr, country=""):
    """Full V3 representation dict for one record (originals preserved)."""
    from .transliterate import transliterate_local
    name_o = "" if name is None else str(name)
    addr_o = "" if addr is None else str(addr)
    nn = normalize_name(name_o)
    aa = normalize_address(addr_o)
    cc = normalize_country(country)
    rec = {
        "name_original": name_o,
        "address_original": addr_o,
        "name_norm": nn,
        "address_norm": aa,
        "country_norm": cc,
        "name_compact": core_name_compact(nn),
        "address_compact": " ".join(aa.split()),
        "name_tokens": nn.split(),
        "address_tokens": aa.split(),
        "name_transliterated": transliterate_local(nn),
        "address_transliterated": transliterate_local(aa),
        "numeric_tokens": extract_numbers(name_o + " " + addr_o),
        "numeric_signature": extract_numeric_signature(name_o + " " + addr_o),
        "postal_candidates": extract_postal_candidates(addr_o),
        "house_numbers": extract_house_numbers(addr_o),
    }
    return rec


def e5_name_view(rec) -> str:
    return ("name: %s address: %s country: %s" % (rec.get("name_norm", ""), "", rec.get("country_norm", ""))).strip()


def e5_full_view(rec) -> str:
    return ("name: %s address: %s country: %s"
            % (rec.get("name_norm", ""), rec.get("address_norm", ""), rec.get("country_norm", ""))).strip()
