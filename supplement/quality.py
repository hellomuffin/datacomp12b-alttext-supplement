"""Shared alt-text quality filter + MetaCLIP matcher for the supplement pipeline.

The failure we measured in the current pool: 22% of alt-texts have <=2 words, 2.9% are
literal placeholders ("Untitled photo", "image", "Thumbnail", ...), 5% are duplicates.
`is_informative` rejects exactly those classes. Thresholds are deliberate choices:
MIN_WORDS=4 because 38% of the pool is <=4 words vs 18% in metaclip_1b.
"""
import re
import ahocorasick

NUM_ENTRIES = 500_000
MIN_WORDS = 4

_PLACEHOLDER = re.compile(
    r"^(untitled( photo| image|)|image|images|img|photo|photos|picture|pictures|thumbnail|thumb|"
    r"no title|front cover|back cover|product details|product image|gallery photo|gallery image|"
    r"dsc[_ -]?\d*|img[_ -]?\d*|image\d*|photo\d*|screenshot|screen shot|unnamed|default|logo|banner|"
    r"avatar|placeholder|icon|cover|slide ?\d*|page ?\d*|figure ?\d*|fig ?\d*|\d+|none|null|n/a|"
    r"loading|click to enlarge|view larger|main image|[\w-]+\.(jpe?g|png|gif|webp|bmp))$")
_WORD = re.compile(r"[A-Za-z0-9]+")
_URLISH = re.compile(r"(https?://|www\.|\.(jpe?g|png|gif|webp)\b)", re.I)


def is_informative(text):
    """Return (ok, reason). reason in {'ok','placeholder','short','urlish'}."""
    t = (text or "").strip()
    low = t.lower()
    if not t or _PLACEHOLDER.match(low):
        return False, "placeholder"
    words = _WORD.findall(t)
    if len(words) < MIN_WORDS:
        return False, "short"
    if _URLISH.search(low) and len(words) < 8:
        return False, "urlish"
    return True, "ok"


# --- MetaCLIP substring matching (metaclip1/substr_matching.py semantics) ---
_PUNCTS = [",", ".", ";", ":", "?", "!", "`"]


def spacing(text):
    s = f" {text} "
    for p in _PUNCTS:
        s = s.replace(p, f" {p} ")
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def build_automaton(metadata):
    A = ahocorasick.Automaton()
    for i, e in enumerate(metadata):
        A.add_word(f" {e} ", i)
    A.make_automaton()
    return A
