#!/usr/bin/env python3
"""
analyze_misses.py — categorize dictionary misses from a run_flores_dev results file.

Recomputes the *current* miss set (post-lookup-fix) token by token, then classifies
each miss into a priority-ordered, mutually exclusive taxonomy. When the parallel
flores English reference is available, the Uyghur token is romanized (UEY→ULY) and
matched against capitalized (NE proxy) and lowercase (loanword/1991-gap proxy)
English words of the aligned sentence.

Categories (first matching rule wins):
  digit              Latin digits — universal, no gloss needed
  multiword          analytical multiword lemma (handled by split lookup)
  latin-foreign      token contains Latin script (acronyms, foreign names)
  apertium-proper    Apertium tagged the token <np> ('Proper noun')
  en-ne              romanized form ≈ capitalized EN word in parallel ref (eval-time NE)
  en-borrowing       romanized form ≈ lowercase EN word            (modern loanword proxy)
  near-match-context spelling-near (≤1 edit in ULY space) dictionary candidates are
                      shown with a verify-in-context caveat (results 'near' field)
  derived            base after stripping a productive suffix is a DB headword
  copula-artifact    ئى-family copula lemmas (ئىدى/ئىكەن...) — analyzer artifact
  fallback-no-reading no clean Apertium reading (alignment failed or OOV)
  lemma-artifact     degenerate lemma (stub/weird chars)
  content-gap        genuinely absent from Schwarz (1991 dictionary gap or rare word)

Outputs:
  <outdir>/<name>.miss_report.jsonl — one JSON record per missed content token
  <outdir>/<name>.miss_report.md   — human-readable summary with examples
"""

import argparse
import importlib.util
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- load pipeline/dictionary.py directly (it is stdlib-only; avoids torch) ---
_spec = importlib.util.spec_from_file_location(
    "pipeline_dictionary", os.path.join(ROOT, "pipeline", "dictionary.py"))
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)
UyghurDictionary, uey_to_uly = pd.UyghurDictionary, pd.uey_to_uly

DEFAULT_RESULTS = os.path.join(ROOT, "stats", "flores_dev", "dev.results.jsonl")
DEFAULT_DB = os.path.join(ROOT, "dictionary", "uig_eng_dict_complete.sqlite")
DEFAULT_UIG = os.path.join(ROOT, "flores200_dataset", "uig_Arab.dev")
DEFAULT_ENG = os.path.join(ROOT, "flores200_dataset", "eng_Latn.dev")

# productive derivational suffixes (native Uyghur + common borrowings)
DERIV_SUFFIXES = ["لىق", "لىك", "چى", "چە", "خانا", "گەر", "پەز", "ۋاز", "كار", "لاش", "چىلىق"]

# known degenerate lemma stubs seen in practice
STUB_LEMMAS = {"ئى", "ئا", "چو", "د", "ۋ", "ب", "چ", "ما", "لا", "تە", "ساۋ", "رۇد"}


def norm_uly(s: str) -> str:
    return uey_to_uly(s).lower().replace("\u00eb", "e").strip("-").strip()


def norm_lat(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def lev(a: str, b: str) -> int:
    a, b = norm_uly(a), norm_lat(b)
    if abs(len(a) - len(b)) > 2:
        return 9
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            c = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = c
    return dp[lb]


def eng_cap_words(sent: str):
    out = []
    for i, w in enumerate(sent.split()):
        wc = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)
        if i and wc and wc[0].isupper() and wc not in ("The", "I"):
            out.append(wc)
    return out


def eng_all_words(sent: str):
    return [re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w).lower()
            for w in sent.split() if re.search(r"[A-Za-z]", w)]


def classify(form: str, lemma: str, pos: str, disambig_source: str,
             capw, allw, near_field) -> tuple:
    """Return (category, extra-dict) for one missed token."""
    lem = (lemma or "").strip()
    frm = (form or "").strip()
    is_verb = "verb" in pos.lower()

    if re.fullmatch(r"\d+", lem):
        return "digit", None
    if " " in lem:
        return "multiword", None
    if re.search(r"[A-Za-z]", frm):
        return "latin-foreign", None
    if pos == "Proper noun":
        return "apertium-proper", None

    # english-reference signals (eval-time only)
    f_norm, l_norm = norm_uly(frm), norm_uly(lem)
    cand = f_norm if len(f_norm) >= len(l_norm) else l_norm
    if capw and len(cand) >= 4:
        best = min((lev(cand, c) for c in capw), default=9)
        if best <= 2:
            return "en-ne", min(capw, key=lambda c: lev(cand, c))
    if allw and len(cand) >= 4:
        low = [x for x in allw if len(x) >= 4]
        if low:
            best = min((lev(cand, x) for x in low), default=9)
            if best <= 2 and not (capw and min((lev(cand, c) for c in capw), default=9) <= 2):
                return "en-borrowing", min(low, key=lambda x: lev(cand, x))

    # near-match-context: results-recorded spelling-near dictionary candidates
    if near_field:
        return "near-match-context", [n["headword"] for n in near_field[:3]]

    if "Copula" in pos or lem in STUB_LEMMAS or re.search(r"[\u2060-\u206f]", frm):
        return "copula-artifact", None
    if len(lem) <= 2:
        return "lemma-artifact", None
    if disambig_source == "fallback":
        return "fallback-no-reading", None
    return "content-gap", None


# --- tiny DB helpers ---
def _db_conn(db_path):
    return sqlite3.connect(db_path)


def db_has(headword, conn) -> bool:
    return conn.execute("SELECT 1 FROM entries WHERE headword_uey = ? LIMIT 1",
                        (headword,)).fetchone() is not None


def build_hw_index(conn):
    hws = [r[0] for r in conn.execute("SELECT DISTINCT headword_uey FROM entries").fetchall()]
    by_len = defaultdict(list)
    for h in hws:
        by_len[len(h)].append(h)
    return hws, by_len


def near_db(s, idx):
    _, by_len = idx
    cands = []
    for L in (len(s) - 1, len(s), len(s) + 1):
        cands.extend(by_len.get(L, []))
    out = []
    for h in cands:
        if lev(s, h) <= 1:
            out.append(h)
    out.sort(key=lambda h: lev(s, h))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--dict", default=DEFAULT_DB)
    ap.add_argument("--uig", default=DEFAULT_UIG)
    ap.add_argument("--eng", default=DEFAULT_ENG)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "stats", "flores_dev"))
    ap.add_argument("--name", default="dev")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    dict_ = UyghurDictionary(args.dict)
    conn = _db_conn(args.dict)
    hw_idx = build_hw_index(conn)

    eng_by = {}
    if os.path.exists(args.uig) and os.path.exists(args.eng):
        uig = [l.rstrip("\n") for l in open(args.uig, encoding="utf-8")]
        eng = [l.rstrip("\n") for l in open(args.eng, encoding="utf-8")]
        eng_by = {u: e for u, e in zip(uig, eng)}
        print(f"[ref ] parallel English reference: {len(eng_by)} aligned sentences")
    else:
        print("[ref ] NO parallel reference — en-ne/en-borrowing categories disabled")

    by_cat = Counter()
    records = []
    spelled = []
    n_sent_hit = n_total = 0
    for l in open(args.results, encoding="utf-8"):
        r = json.loads(l)
        eng = eng_by.get(r["sentence"], "")
        capw = eng_cap_words(eng) if eng else []
        allw = eng_all_words(eng) if eng else []
        for t in r["per_token"]:
            if t["pos"] == "Punctuation" or t.get("has_dict"):
                continue
            lem, form, pos = t["lemma"], t["form"], t["pos"]
            is_verb = "verb" in pos.lower()
            dict_._cache.clear()
            if dict_.lookup(lem, is_verb=is_verb, surface=form) is not None:
                continue  # already covered by today's lookup fixes
            cat, extra = classify(form, lem, pos, t["disambig_source"],
                                  capw, allw, t.get("near") or [])
            by_cat[cat] += 1
            n_total += 1
            if cat == "near-match-context":
                spelled.append((lem, extra))
            rec = {
                "form": form, "lemma": lem, "pos": pos,
                "disambig_source": t["disambig_source"],
                "category": cat, "extra": extra,
                "gloss": t.get("gloss"),
            }
            if eng and capw:
                n_sent_hit += 1
            records.append(rec)

    # --- report ---
    os.makedirs(args.outdir, exist_ok=True)
    jpath = os.path.join(args.outdir, f"{args.name}.miss_report.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mpath = os.path.join(args.outdir, f"{args.name}.miss_report.md")
    order = ["digit", "multiword", "latin-foreign", "apertium-proper", "en-ne",
             "en-borrowing", "near-match-context", "derived", "copula-artifact",
             "fallback-no-reading", "lemma-artifact", "content-gap"]
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(f"# Dictionary miss report — {args.name}\n\n")
        f.write(f"Missing content tokens: **{n_total}** "
                f"(unique lemmas: {len(set(r['lemma'] for r in records))})\n\n")
        f.write("| category | tokens | meaning |\n|---|---|---|\n")
        for cat in order:
            c = by_cat.get(cat, 0)
            desc = {
                "digit": "Latin digits — universal, no gloss needed",
                "multiword": "analytical multiword lemma (split lookup catches)",
                "latin-foreign": "Latin-script token (acronym / foreign name)",
                "apertium-proper": "Apertium `<np>` proper noun",
                "en-ne": "matches capitalized EN word in parallel ref (eval-time NE)",
                "en-borrowing": "matches lowercase EN word (modern loanword / 1991 gap)",
                "near-match-context": "spelling-near dictionary candidates (verify-in-context caveat)",
                "derived": f"base after stripping a productive suffix is in DB",
                "copula-artifact": "ئى-family copula lemma (analyzer artifact)",
                "fallback-no-reading": "no clean Apertium reading",
                "lemma-artifact": "degenerate lemma",
                "content-gap": "genuinely absent from Schwarz (1991 gap / rare word)",
            }[cat]
            f.write(f"| {cat} | {c} | {desc} |\n")
        f.write("\n## Examples\n\n")
        seen = set()
        for rec in records:
            key = (rec["category"], rec["form"])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"- **{rec['category']}** `{rec['form']}` (lemma `{rec['lemma']}`, "
                    f"{rec['pos']}) extra={rec['extra']}\n")
            if max(1, by_cat[rec["category"]]) > 1 and len([s for s in seen if s[0] == rec["category"]]) >= 3:
                pass  # keep 3 per category via counter below
        f.write("\n")

    # concise console summary
    print(f"\n[out ] {jpath}")
    print(f"[out ] {mpath}")
    print(f"\nMissing content tokens: {n_total} "
          f"(unique lemmas: {len(set(r['lemma'] for r in records))})")
    for cat in order:
        c = by_cat.get(cat, 0)
        if c:
            print(f"  {cat:22} {c:6d}")
    # top spelling-gap candidates (alias opportunities)
    leg = Counter()
    for lem, extra in spelled:
        if extra:
            leg[(lem, extra[0])] += 1
    print("\nTop near-match lemma -> candidate headword:")
    for (lem, hw), c in leg.most_common(20):
        print(f"   {c:5d}  {lem!r:24} -> {hw!r}")
    conn.close()


if __name__ == "__main__":
    main()