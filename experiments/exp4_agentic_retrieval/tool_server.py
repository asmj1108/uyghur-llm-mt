"""
tool_server.py - Exp 4: serves the pipeline's linguistic information as
agent-callable tools, against the persisted per-sentence analysis
(stats/flores_dev/dev.results.jsonl).

Design (see DESIGN.md §4):
- Minimal units: one information type per tool.
- Purity invariant: nothing analytical outside a tool call.
- Fallbacks opt-in: misses are clean, exploration is the agent's decision.
- Static-prompt parity: returns reuse the exact rendering functions of the
  push prompts (prompt_variants.py).
- No compromises: unresolvable/malformed queries get honest errors; every
  non-exact resolution is logged as data.

Runs in the vLLM env (stdlib + prompt_variants only; no GPU).
"""

import json
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from experiments.common.prompt_variants import (  # noqa: E402
    FLORES_RESULTS_PATH,
    _clause_lines,
    _morph_summary,
    uey_to_uly,
)

_POS_WORDS = {
    "n.": "noun", "v.": "verb", "vt.": "transitive verb",
    "vi.": "intransitive verb", "vt., vi.": "verb", "adj.": "adjective",
    "adv.": "adverb", "pro.": "pronoun", "num.": "numeral",
    "pp.": "postposition", "conj.": "conjunction", "intj.": "interjection",
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u0640"))

# Short English glosses for the frequent UD relations in the MUDT analyses
# of FLORES dev (punct/amod/nmod/nsubj/obj/conj/root/advmod/advcl/case:*/cc/
# nummod/det/fixed/cop/aux/post cover ~98% of all arcs). Raw deprel fallback.
_DEPREL_GLOSS = {
    "root": "main predicate of the clause",
    "nsubj": "subject", "csubj": "subject",
    "obj": "object", "iobj": "indirect object",
    "obl": "oblique phrase",
    "advcl": "adverbial clause",
    "advmod": "adverbial modifier",
    "nmod": "noun modifier", "amod": "adjectival modifier",
    "nummod": "numeral modifier",
    "det": "determiner",
    "case": "case marker", "case:poss": "possessive case marker",
    "case:loc": "locative case marker", "case:dat": "dative case marker",
    "case:abl": "ablative case marker",
    "cc": "coordinating conjunction", "conj": "conjoined element",
    "aux": "auxiliary verb", "cop": "copula", "cop:zero": "copula",
    "post": "postposition", "instr:post": "postposition",
    "fixed": "fixed multiword expression",
    "punct": "punctuation",
    "appos": "apposition", "discourse": "discourse element",
}


def _rel_gloss(rel: str) -> str:
    """'obj' -> 'obj (object)'; unmapped relations pass through unchanged."""
    gloss = _DEPREL_GLOSS.get(rel) or _DEPREL_GLOSS.get(rel.split(":")[0])
    return f"{rel} ({gloss})" if gloss else rel


def _norm(s: str) -> str:
    """NFC + strip zero-width/tatweel — guards Arabic-script echo corruption."""
    s = unicodedata.normalize("NFC", s)
    return s.translate(_ZERO_WIDTH).strip()


def _lev1(a: str, b: str) -> bool:
    """True if Levenshtein distance(a, b) <= 1 (short strings, early exit)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) > len(b):
        a, b = b, a
    # a is shorter or equal
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            if diffs > 1:
                return False
            if len(a) == len(b):
                i += 1
                j += 1
            else:
                j += 1
    return diffs + (len(b) - j) <= 1


def _parse_gloss(gloss: str) -> Tuple[str, str, str]:
    """'kün (n.): 1. Sun / 2. Day' -> ('kün', 'noun', '1. Sun / 2. Day')."""
    head, sep, senses = gloss.partition(": ")
    if not sep:
        return gloss.strip(" ()"), "", ""
    m = re.search(r"\(([^()]*)\)\s*$", head)
    pos_raw = m.group(1) if m else ""
    uly = re.sub(r"\s*\([^()]*\)\s*$", "", head).strip()
    return uly, _POS_WORDS.get(pos_raw, pos_raw), senses


# ---------------------------------------------------------------------------
# Tool schemas (clear descriptions are a first-class requirement: the LLM
# must know exactly what input each function takes and what it returns)
# ---------------------------------------------------------------------------
_WORD_PARAM = {
    "type": "string",
    "description": "One Uyghur word, copied exactly as it appears in the "
                   "source sentence.",
}
_ID_PARAM = {
    "type": "integer",
    "description": "Optional alternative to 'word': the word's position "
                   "number in the sentence, as listed by get_word_list.",
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_word_list",
            "description": (
                "Takes no arguments. Returns the source sentence as a numbered "
                "list of its words (token numbers, forms only, no analysis). "
                "Use the token numbers to address a specific word in other "
                "tools via the token_id parameter."),
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gloss",
            "description": (
                "Looks up ONE word of the source sentence in an Uyghur-English "
                "dictionary. Input: the word (copy it exactly), or its token "
                "number from get_word_list as token_id. Returns: the word's "
                "dictionary headword (lemma) with Latin transliteration, and "
                "its numbered English senses (e.g. '1. Sun / 2. Day'), or a "
                "notice that the word has no dictionary entry. Call this for "
                "any word whose meaning you are not sure of."),
            "parameters": {"type": "object",
                           "properties": {"word": _WORD_PARAM,
                                          "token_id": _ID_PARAM},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lemma",
            "description": (
                "Returns the dictionary headword (lemma/stem) of ONE word of "
                "the source sentence, with its Latin transliteration — for "
                "example 'word X -> lemma Y (y)'. Input: the word (copy it "
                "exactly), or its token number as token_id. Unlike get_gloss "
                "this returns no English meaning; use it to recognize the "
                "stem or its Latin form (e.g. to spot international loanwords "
                "like 'uniwërsitët'), or when a get_gloss lookup failed."),
            "parameters": {"type": "object",
                           "properties": {"word": _WORD_PARAM,
                                          "token_id": _ID_PARAM},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_near_matches",
            "description": (
                "For ONE word of the source sentence that has no exact "
                "dictionary entry: returns up to 3 headwords that are "
                "spelling-similar to it, each with its part of speech and "
                "English senses, marked '≈'. These candidates are NOT "
                "verified — some may be wrong words that merely look similar; "
                "only use one if its meaning fits the sentence. Input: the "
                "word (copy exactly), or its token number as token_id."),
            "parameters": {"type": "object",
                           "properties": {"word": _WORD_PARAM,
                                          "token_id": _ID_PARAM},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_features",
            "description": (
                "Returns the grammatical analysis of ONE word of the source "
                "sentence: its part of speech and morphological features "
                "(case, number, possessor, tense, etc.). Input: the word "
                "(copy exactly), or its token number as token_id. Useful for "
                "words whose meaning you know but whose grammatical role you "
                "want to confirm."),
            "parameters": {"type": "object",
                           "properties": {"word": _WORD_PARAM,
                                          "token_id": _ID_PARAM},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_syntax",
            "description": (
                "Takes no arguments. Returns a one-line summary of the "
                "sentence's clause structure: the main predicate, the "
                "subjects, the objects, and the adverbial/oblique phrases, "
                "each with its dependency role. Call this once if you are "
                "unsure about the sentence's structure."),
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
]

TOOLS_BY_NAME = {t["function"]["name"]: t for t in TOOLS_SCHEMA}
LEAN_TOOLS = [TOOLS_BY_NAME[n] for n in ("get_word_list", "get_gloss")]

# E4-W follow-up (see DESIGN_word_syntax.md): the same six tools, but
# get_syntax is word-addressed — local dependency context per word instead
# of the whole-sentence clause digest. get_syntax REPLACES (not joins) the
# sentence-level tool: tool count stays 6 (E4-A vs E4-B showed more tools
# hurt Gemma).
WORD_SYNTAX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_syntax",
            "description": (
                "Takes ONE word of the source sentence (copy it exactly), or "
                "its token number as token_id. Returns that word's place in "
                "the sentence's dependency structure: its grammatical "
                "relation, the word it depends on (and that word's own "
                "role), and its own dependents with their relations. Use "
                "this to see how a word fits structurally — for example "
                "which verb a modifier or an argument belongs to."),
            "parameters": {"type": "object",
                           "properties": {"word": _WORD_PARAM,
                                          "token_id": _ID_PARAM},
                           "additionalProperties": False},
        },
    }
    if t["function"]["name"] == "get_syntax" else t
    for t in TOOLS_SCHEMA
]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class ToolServer:
    """Executes tool calls against the persisted per-sentence analysis."""

    def __init__(self, records: List[dict],
                 syntax_granularity: str = "sentence"):
        self.records = records  # dev.results.jsonl records (per_token lists)
        # "sentence": get_syntax() -> whole-sentence clause digest (E4-A)
        # "word":      get_syntax(word) -> local dependency context (E4-W)
        self.syntax_granularity = syntax_granularity

    # ---- word resolution ---------------------------------------------------
    def resolve(self, sent_idx: int, word: Optional[str],
                token_id=None) -> Tuple[Optional[dict], str, list]:
        """Returns (token, resolution_path, notes)."""
        toks = self.records[sent_idx]["per_token"]
        notes = []
        if token_id is not None and str(token_id).strip() != "":
            try:  # JSON args often arrive as strings
                tid = int(str(token_id).strip())
            except ValueError:
                tid = None
            if tid is not None:
                for t in toks:
                    if t["id"] == tid:
                        return t, "token_id", notes
                return None, "token_id_invalid", notes

        if not word or not _norm(word):
            return None, "no_input", notes

        nw = _norm(word)
        exact = [t for t in toks if _norm(t["form"]) == nw]
        if exact:
            lemmas = {t["lemma"] for t in exact}
            if len(exact) > 1 and len(lemmas) > 1:
                notes.append(f"appears {len(exact)} times with different "
                             f"analyses; first occurrence returned "
                             f"(use token_id to address a specific one)")
            return exact[0], "exact", notes

        near = [t for t in toks if _lev1(_norm(word), _norm(t["form"]))]
        if len(near) == 1:
            notes.append(f"query '{word}' did not match exactly; resolved to "
                         f"the nearest word '{near[0]['form']}'")
            return near[0], "edit_distance_1", notes
        if len(near) > 1:
            amb = ", ".join(t["form"] for t in near[:5])
            return None, "ambiguous", [f"could match several words: {amb}"]

        # stem-prefix resolution (agglutinative morphology: the agent may
        # query the stem of an inflected surface form)
        if len(nw) >= 3:
            pref = [t for t in toks
                    if _norm(t["form"]).startswith(nw) and t["pos"] != "Punctuation"]
            if len(pref) == 1:
                notes.append(f"query '{word}' resolved as the stem of "
                             f"'{pref[0]['form']}'")
                return pref[0], "stem_prefix", notes
            if len(pref) > 1:
                amb = ", ".join(t["form"] for t in pref[:5])
                return None, "ambiguous", [f"stem of several words: {amb}"]

        return None, "unresolved", notes

    def _word_list(self, toks) -> str:
        return "  ".join(f"{t['id']}. {t['form']}" for t in toks)

    # ---- execution -----------------------------------------------------------
    def execute(self, sent_idx: int, name: str, args: dict) -> Tuple[str, dict]:
        """Run one tool call. Returns (return_text, structured_log)."""
        rec = self.records[sent_idx]
        toks = rec["per_token"]
        log = {"tool": name, "args": args, "resolved": None,
               "resolution": None, "result": None}

        if name == "get_word_list":
            log["result"] = "word_list"
            return self._word_list(toks), log

        if name == "get_syntax" and self.syntax_granularity == "sentence":
            lines = _clause_lines(toks, [], use_morph=False, use_dict=False)
            log["result"] = "syntax"
            return "\n".join(lines), log

        # word-addressed tools
        word = args.get("word")
        token_id = args.get("token_id")
        tok, path, notes = self.resolve(sent_idx, word, token_id)
        log["resolution"] = path
        if tok is not None:
            log["resolved"] = tok["id"]
        log["notes"] = notes

        if tok is None:
            if path == "token_id_invalid":
                log["result"] = "error_bad_token_id"
                return (f"No word with token number {token_id} in this "
                        f"sentence. Words: {self._word_list(toks)}"), log
            if path in ("no_input", "unresolved", "ambiguous"):
                log["result"] = "error_unresolved"
                hint = ("\nProvide either 'word' (one word copied exactly from "
                        "the sentence) or 'token_id'. "
                        f"Words in the sentence: {self._word_list(toks)}")
                return (f"Could not match any word of the sentence.{hint}"), log

        t = tok
        if t["pos"] == "Punctuation":
            log["result"] = "punctuation"
            return (f"Token #{t['id']}: {t['form']} — punctuation; no "
                    f"dictionary analysis."), log

        surf_uly = uey_to_uly(t["form"]) or t["uly"]
        header = f"Token #{t['id']}: {t['form']} ({surf_uly})"

        if name == "get_lemma":
            log["result"] = "lemma"
            return (f"{header} → lemma {t['lemma']} ({t['uly']})"), log

        if name == "get_gloss":
            if t["has_dict"]:
                uly, pos_word, senses = _parse_gloss(t["gloss"])
                log["result"] = "hit"
                lines = [header,
                         f"Lemma: {t['lemma']} ({t['uly']})"
                         + (f" — {pos_word}" if pos_word else "")]
                if senses:
                    lines.append(f"Senses: {senses}")
                else:  # synthetic/universal entries (digits, Latin)
                    lines.append(f"Entry: {t['gloss']}")
                return "\n".join(lines), log
            log["result"] = "miss"
            return (f"{header} — no dictionary entry.\n(Options: get_lemma "
                    f"for the stem/transliteration, get_near_matches for "
                    f"spelling-near candidates.)"), log

        if name == "get_near_matches":
            near = t.get("near") or []
            if t["has_dict"]:
                log["result"] = "has_exact_entry"
                return (f"{header} — this word HAS an exact dictionary entry; "
                        f"no near-match search needed (use get_gloss)."), log
            if not near:
                log["result"] = "no_candidates"
                return (f"{header} — no spelling-near candidates found."), log
            log["result"] = f"candidates_{len(near[:3])}"
            lines = [f"{header} — no exact entry. Spelling-near candidates "
                     f"(NOT verified — check that the meaning fits the "
                     f"sentence before using one):"]
            lines += [f"≈ {n['gloss']}" for n in near[:3]]
            return "\n".join(lines), log

        if name == "get_features":
            log["result"] = "features"
            return f"{header} — {_morph_summary(t)}", log

        if name == "get_syntax":  # word granularity (E4-W)
            log["result"] = "syntax_word"
            return self._word_syntax(toks, t), log

        log["result"] = "unknown_tool"
        return (f"Unknown tool '{name}'. Available tools: "
                f"{', '.join(TOOLS_BY_NAME)}"), log


    def _word_syntax(self, toks: List[dict], t: dict) -> str:
        """E4-W: local dependency context for one resolved token.

        Presents the queried word's arc to its head AND the head's own
        deprel (one extra level of clausal context — knowing the head is
        `root` vs `advcl` tells the model whether it is inside a
        subordinate clause), plus each dependent's arc. No transitive
        head chain: full-tree knowledge costs O(N) calls by design
        (DESIGN_word_syntax.md §2).
        """
        by_id = {x["id"]: x for x in toks}
        deps = [d for d in toks
                if d["head_id"] == t["id"] and d["id"] != t["id"]]
        lines = []
        if t["head_id"] == 0:
            lines.append(f"Token #{t['id']}: {t['form']} — "
                         f"root ({_DEPREL_GLOSS['root']})")
        else:
            h = by_id.get(t["head_id"])
            lines.append(f"Token #{t['id']}: {t['form']}")
            if h is None:
                lines.append(f"head: #{t['head_id']} (not available) — "
                             f"{t['form']} is {_rel_gloss(t['deprel'])} "
                             f"of it")
            else:
                head_role = (f"{h['form']} is {_rel_gloss(h['deprel'])}"
                             if h["deprel"] != "root"
                             else f"{h['form']} is root "
                                  f"({_DEPREL_GLOSS['root']})")
                lines.append(f"head: #{h['id']} {h['form']} — "
                             f"{t['form']} is {_rel_gloss(t['deprel'])} "
                             f"of it; {head_role}")
        if deps:
            dep_s = ", ".join(f"#{d['id']} {d['form']} ({d['deprel']} "
                               f"of {t['form']})" for d in deps)
            lines.append(f"dependents: {dep_s}")
        else:
            lines.append("dependents: (none)")
        return "\n".join(lines)


def load_server(records_path: str = FLORES_RESULTS_PATH,
                syntax_granularity: str = "sentence") -> ToolServer:
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return ToolServer(records, syntax_granularity=syntax_granularity)
