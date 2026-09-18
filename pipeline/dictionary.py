"""
dictionary.py - Dictionary lookup module using Henry G. Schwarz Uyghur-English SQLite Database
"""

import sqlite3
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def _norm_uly(s: str) -> str:
    """Romanized lowercase form used for spelling-near scoring (ë collapsed to e)."""
    return uey_to_uly(s).lower().replace('ë', 'e').strip('-').strip()



UEY_TO_ULY_MAP = {
    'ئا': 'a', 'ا': 'a',
    'ئە': 'e', 'ە': 'e',
    'ئې': 'ë', 'ې': 'ë',
    'ئى': 'i', 'ى': 'i',
    'ئو': 'o', 'و': 'o',
    'ئۇ': 'u', 'ۇ': 'u',
    'ئۆ': 'ö', 'ۆ': 'ö',
    'ئۈ': 'ü', 'ۈ': 'ü',
    'ب': 'b', 'پ': 'p', 'ت': 't', 'ج': 'j', 'چ': 'ch', 'خ': 'x', 'د': 'd',
    'ر': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'غ': 'gh', 'ف': 'f',
    'ق': 'q', 'ك': 'k', 'گ': 'g', 'ڭ': 'ng', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ھ': 'h', 'ۋ': 'w', 'ي': 'y', 'ئ': ''
}

DEVOICING_PAIRS = {
    'ب': 'پ', 'پ': 'ب',
    'گ': 'ك', 'ك': 'گ',
    'غ': 'ق', 'ق': 'غ',
    'د': 'ت', 'ت': 'د',
    'ج': 'چ', 'چ': 'ج',
    'ز': 'س', 'س': 'ز'
}



def uey_to_uly(word: str) -> str:
    """Convert Uyghur Arabic Script (UEY) word to Standard Uyghur Latin (ULY)."""
    res = []
    i = 0
    w = word.strip()
    while i < len(w):
        if i + 1 < len(w) and w[i:i + 2] in UEY_TO_ULY_MAP:
            res.append(UEY_TO_ULY_MAP[w[i:i + 2]])
            i += 2
        elif w[i] in UEY_TO_ULY_MAP:
            res.append(UEY_TO_ULY_MAP[w[i]])
            i += 1
        else:
            res.append(w[i])
            i += 1
    return ''.join(res)


def _lev_banded(a: str, b: str, k: int = 1) -> int:
    """Edit distance between two PRE-NORMALIZED strings, returning k+1 when > k.

    Computes only the band |i-j| <= k and exits as soon as the whole row exceeds
    k — the common unrelated-pair case costs O(len).
    """
    if abs(len(a) - len(b)) > k:
        return k + 1
    if a == b:
        return 0
    la, lb = len(a), len(b)
    row = {j: j for j in range(min(k, lb) + 1)}
    for i in range(1, la + 1):
        ndp = {}
        lo, hi = max(0, i - k), min(lb, i + k)
        for j in range(lo, hi + 1):
            if j == 0:
                ndp[0] = i
            else:
                dele = row.get(j, k + 1) + 1
                subs = row.get(j - 1, k + 1) + (a[i - 1] != b[j - 1])
                inser = ndp.get(j - 1, k + 1) + 1
                ndp[j] = min(dele, subs, inser)
        if min(ndp.values()) > k:
            return k + 1
        row = ndp
    return row.get(lb, k + 1)


@dataclass
class DictEntry:
    """Represents a clean dictionary entry tailored for LLM prompt context."""
    headword_uey: str
    headword_uly: str
    pos: Optional[str]
    definitions: List[str] = field(default_factory=list)
    domain: Optional[str] = None

    @property
    def short_gloss(self) -> str:
        """Return a concise, clean single-line English gloss."""
        pos_str = f" ({self.pos})" if self.pos else ""
        if not self.definitions:
            return f"{self.headword_uly}{pos_str}"
        
        # Clean up definitions (remove trailing periods, duplicate semicolons)
        cleaned_defs = []
        for d in self.definitions[:3]:
            cd = d.strip().rstrip('.')
            if cd and cd not in cleaned_defs:
                cleaned_defs.append(cd)
        defs_str = "; ".join(cleaned_defs)
        return f"{self.headword_uly}{pos_str}: {defs_str}"


class UyghurDictionary:
    """
    Uyghur-English Dictionary interface to SQLite database.
    Optimized for high-precision morphological stem lookup with phonetic fallback.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        self._cache: Dict[str, Optional[DictEntry]] = {}
        self._hw_index: Optional[Dict[int, List[str]]] = None
        self._hw_norm: Optional[Dict[str, str]] = None
        self._near_cache: Dict[str, List[DictEntry]] = {}

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    @staticmethod
    def generate_stem_variants(stem: str, is_verb: bool = False) -> List[str]:
        """
        Generate candidate spelling variants for dictionary lookup:
        - Verb hyphen variations (e.g. 'كەل-' vs 'كەل')
        - Terminal devoicing variations (e.g. 'كىتاب' vs 'كىتاپ')
        """
        base = stem.strip().rstrip('-')
        variants = []

        if is_verb:
            variants.extend([base + '-', base])
        else:
            variants.extend([base, base + '-'])

        if base and base[-1] in DEVOICING_PAIRS:
            alt_base = base[:-1] + DEVOICING_PAIRS[base[-1]]
            if is_verb:
                variants.extend([alt_base + '-', alt_base])
            else:
                variants.extend([alt_base, alt_base + '-'])

        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique_variants.append(v)
        return unique_variants

    def lookup(
        self,
        stem: str,
        is_verb: bool = False,
        pos_hint: Optional[str] = None,
        surface: Optional[str] = None
    ) -> Optional[DictEntry]:
        """
        Lookup a word stem in the dictionary.

        `surface` is the surface form of the token: used as a last-resort
        fallback when the stem itself is not a dictionary headword.
        """
        clean_stem = stem.strip()
        if not clean_stem:
            return None

        cache_key = f"{clean_stem}_{is_verb}_{pos_hint}_{surface}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cur = self.conn.cursor()

        # 0. Latin numerals are universal -- the model needs no gloss.
        if re.fullmatch(r"\d+", clean_stem):
            entry = DictEntry(
                headword_uey=clean_stem,
                headword_uly=clean_stem,
                pos="num.",
                definitions=[],
            )
            self._cache[cache_key] = entry
            return entry

        # 0a. Latin-script words are universal (acronyms, proper names, code) --
        # the model needs no dictionary gloss.
        if re.search(r"[A-Za-z]", clean_stem):
            entry = DictEntry(
                headword_uey=clean_stem,
                headword_uly=uey_to_uly(clean_stem) or clean_stem,
                pos="foreign",
                definitions=[],
            )
            self._cache[cache_key] = entry
            return entry

        # 0b. Analytical multiword lemmas (e.g. 'تەلەپ قىل'): look up the
        # component words individually.
        if ' ' in clean_stem:
            for part in clean_stem.split():
                entry = self.lookup(part, is_verb=False)
                if entry is not None:
                    self._cache[cache_key] = entry
                    return entry
            self._cache[cache_key] = None
            return None

        variants = self.generate_stem_variants(clean_stem, is_verb)

        # 1. Search by Arabic script headword variants
        for v in variants:
            cur.execute("""
                SELECT e.id, e.headword_uey, e.headword_uly, e.pos, e.domain, s.definition
                FROM entries e
                LEFT JOIN senses s ON e.id = s.entry_id
                WHERE e.headword_uey = ?
                ORDER BY e.id, s.sense_number
            """, (v,))
            rows = cur.fetchall()
            if rows:
                entry = self._build_dict_entry(rows)
                self._cache[cache_key] = entry
                return entry

        # 2. Search by ULY transliteration fallback
        uly_stem = uey_to_uly(clean_stem)
        uly_variants = [uly_stem, uly_stem + '-', uly_stem.rstrip('-')]
        for uv in uly_variants:
            cur.execute("""
                SELECT e.id, e.headword_uey, e.headword_uly, e.pos, e.domain, s.definition
                FROM entries e
                LEFT JOIN senses s ON e.id = s.entry_id
                WHERE e.headword_uly = ?
                ORDER BY e.id, s.sense_number
            """, (uv,))
            rows = cur.fetchall()
            if rows:
                entry = self._build_dict_entry(rows)
                self._cache[cache_key] = entry
                return entry

        # 3. Surface-form fallback: the stem missed, but the full surface
        # form is itself a dictionary headword (e.g. stem 'ئالد', form 'ئالدى').
        if surface and surface.strip() and surface.strip() != clean_stem:
            for v in self.generate_stem_variants(surface.strip(), is_verb):
                cur.execute("""
                    SELECT e.id, e.headword_uey, e.headword_uly, e.pos, e.domain, s.definition
                    FROM entries e
                    LEFT JOIN senses s ON e.id = s.entry_id
                    WHERE e.headword_uey = ?
                    ORDER BY e.id, s.sense_number
                """, (v,))
                rows = cur.fetchall()
                if rows:
                    entry = self._build_dict_entry(rows)
                    self._cache[cache_key] = entry
                    return entry

        self._cache[cache_key] = None
        return None

    def _near_index(self) -> Dict[int, List[str]]:
        """Lazily built index: normalized-ULY length -> headwords, for near search."""
        if self._hw_index is None:
            idx: Dict[int, List[str]] = {}
            norms: Dict[str, str] = {}
            for (hw,) in self.conn.execute("SELECT DISTINCT headword_uey FROM entries"):
                n = _norm_uly(hw)
                norms[hw] = n
                idx.setdefault(len(n), []).append(hw)
            self._hw_index = idx
            self._hw_norm = norms
        return self._hw_index

    def lookup_near(
        self,
        stem: str,
        surface: Optional[str] = None,
        limit: int = 3
    ) -> List[DictEntry]:
        """
        Spelling-near candidates for a stem that has no exact dictionary entry.

        Finds DB headwords within edit distance 1 of the stem (and the surface
        form, when given) in normalized ULY space. Returns up to `limit`
        entries so the caller can present the candidate glosses to the LLM with
        a verify-in-context caveat (they are close spellings, not assertions).
        """
        stem = (stem or "").strip()
        if not stem or " " in stem:
            return []
        cache_key = f"{stem}|{surface or ''}"
        if cache_key in self._near_cache:
            return self._near_cache[cache_key]
        index = self._near_index()   # builds _hw_index and _hw_norm lazily
        norms = self._hw_norm
        candidates: Dict[str, int] = {}
        for src in (stem, surface):
            if not src or not src.strip():
                continue
            s = _norm_uly(src)
            if not s:
                continue
            for L in (max(0, len(s) - 1), len(s), len(s) + 1):
                for hw in index.get(L, ()):
                    if hw in candidates or hw == stem or hw == stem + "-":
                        continue
                    d = _lev_banded(s, norms[hw])
                    if d <= 1:
                        candidates[hw] = d
        ranked = sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]))[:limit]
        entries = []
        for hw, _d in ranked:
            rows = self.conn.execute("""
                SELECT e.id, e.headword_uey, e.headword_uly, e.pos, e.domain, s.definition
                FROM entries e
                LEFT JOIN senses s ON e.id = s.entry_id
                WHERE e.headword_uey = ?
                ORDER BY e.id, s.sense_number
            """, (hw,)).fetchall()
            if rows:
                entries.append(self._build_dict_entry(rows))
        self._near_cache[cache_key] = entries
        return entries

    def _build_dict_entry(self, rows: List[Tuple]) -> DictEntry:
        """Assemble rows from SQLite into a structured DictEntry."""
        headword_uey = rows[0][1]
        headword_uly = rows[0][2]
        pos = rows[0][3]
        domain = rows[0][4]

        definitions = []
        for r in rows:
            defn = r[5]
            if defn and defn.strip() and defn not in definitions:
                # Remove Markdown bold asterisks from legacy OCR formatting
                clean_defn = re.sub(r"\*\*([^*]+)\*\*", r"\1", defn).strip()
                definitions.append(clean_defn)

        return DictEntry(
            headword_uey=headword_uey,
            headword_uly=headword_uly,
            pos=pos,
            definitions=definitions,
            domain=domain
        )

    def lookup_batch(
        self,
        stems_and_verbs: List[Tuple[str, bool]]
    ) -> List[Optional[DictEntry]]:
        """Batch lookup a list of (stem, is_verb) pairs."""
        return [self.lookup(stem, is_verb) for stem, is_verb in stems_and_verbs]

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
