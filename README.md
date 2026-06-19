# Thesis Progress — Memory Protocol

**Topic**: Improving machine translation (Uyghur → English) for a low-resource language via in-context learning with LLMs

---

## Goal

Build a pipeline that extracts linguistic information for an input sentence and presents it as structured context in a prompt to guide an LLM to produce better translations.

---

## Pipeline Overview

```
Input sentence (Uyghur)
        ↓
[1] Dependency Parser     →  HEAD, DEPREL
[2] Morphological Tagger  →  LEMMA, XPOS, FEATS
[3] Dictionary Lookup     →  word-level translations
        ↓
Linguistic description (in-context prompt)
        ↓
LLM → English translation
```

---

## Component Status

### 1. Dependency Parser

- **Data**: [MUDT](https://github.com/wyqmath/MUDT) - Improved UDT
- **Framework**: DiaParser
- **Output**: HEAD, DEPREL
- **Status**: Training script provided by MUDT; model trained successfully

---

### 2. Morphological Tagger

**Tool**: [`apertium-uig`](https://github.com/apertium/apertium-uig) — morphological analyser/generator + Constraint Grammar (CG) tagger for Uyghur

**Problem**: The `uig-tagger` mode uses CG for disambiguation, but the grammar rules are sparse → some tokens retain **multiple readings** after CG

#### Data available for disambiguation

[UyUDT](https://github.com/UniversalDependencies/UD_Uyghur-UDT) — Uyghur UD Treebank in CoNLL-U format:

| Field | Source                                                                 |
|---|------------------------------------------------------------------------|
| UPOS, HEAD, DEPREL | Manual annotation                                                      |
| LEMMA, XPOS, FEATS | Assigned programmatically (via `apertium-uig` + [`conllu-morph.py`](https://github.com/ftyers/ud-scripts/blob/master/conllu-morph.py)) |

#### Approach A: Disambiguate by UPOS (proposed)

- Train a UPOS tagger on UyUDT
- Use predicted UPOS to filter Apertium readings

**Validation test** (oracle upper bound using gold UPOS):
```
For every word in UyUDT with > 1 Apertium reading after CG:
    apply gold UPOS tag as filter
    check if the token is still ambiguous
```
- **Action**: Wrote an N:M monotonic alignment script to map Apertium output to CoNLL-U tokens (achieved >99% pipeline alignment compatibility).
- **Result**: Tested the UPOS tag filter hypothesis across the entire UyUDT dataset (16,866 ambiguous words).
  - Ambiguity Reduced: Only ~25%
  - Still ambiguous: ~74%
  - Word lost entirely (Overfiltered): ~12%
- **Conclusion**: UPOS filtering is definitively insufficient for Uyghur morphological disambiguation. UPOS is too coarse to resolve internal morphological ambiguity (e.g., Noun vs. Noun readings), and pipeline tokenization conflicts lead to unacceptably high data loss (12%).
- **Decision**: Abandoning standalone UPOS filtering.


#### Approach B: Disambiguate by DEPREL (used in UyUDT, for reference)

**Method** (see [`ud-scripts/conllu-morph.py`](ud-scripts/conllu-morph.py) + [`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv)):
- Score each candidate reading by:
    1. Overlap between candidate tags and existing annotation
    2. Co-occurrence frequency of the token's DEPREL with each FEAT value (from `uig-feat-rel.tsv`)
- Select highest-scoring reading

**Limitations**:
- Statistical heuristic — origin of frequency counts in [`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv) is unknown
- Not linguistically/grammatically motivated
- Sub-optimal as a general solution

#### Approach C: Disambiguate using LLM

**Concept**: Large LLMs have deep knowledge of Universal Dependencies from multilingual pre-training. Use a Large LLM to generate high-quality **"Pseudo-Gold"** labels by selecting the correct reading for each ambiguous word, then fine-tune a smaller model for fast, production-ready pipeline inference.

**Architecture**: Neuro-Symbolic Hybrid Pipeline — deterministic syntactic rules resolve what they can with 100% confidence; the LLM (and later, the fine-tuned model) handles only the semantic residual.

* * *

##### **C.1 — Tag-to-Gloss Formatting**

*   Wrote a Python script using `streamparser` to intercept complex Apertium readings (e.g., `<v><tv><ger><nom>`) and translate them into human-readable UD-style glosses via a `.tsv` map.

*   Handles `+`-joined lexical units (e.g., enclitic copula `+ئى<cop>`) by splitting and glossing each part.

*   **Output**: structured candidate descriptions ready for LLM prompting.


* * *

##### **C.2 — Key Discovery: Human Annotation Bias in UyUDT**

Testing revealed that human annotators of UyUDT sometimes violate strict UD principles due to **traditional Turkic grammar biases**.

*   **Example**: `تۇغقاننىڭ` ("relative's") was tagged `ADJ` by humans (because of the `-قان` participial suffix), but is a **fully lexicalized `NOUN`** under UD. The LLM correctly identifies the noun reading.

*   **Implication**: The LLM can produce labels that are **superior to the human baseline** for historically tricky morpho-syntactic cases. This strengthens the motivation for LLM relabeling (vs. trusting gold UPOS as in Approach A).

*   **Constraint**: This same bias means syntax-tree heuristics that *rely on* human UPOS (e.g., Noun-vs-Substantivized-Adjective) are unsafe and **must** be deferred to the LLM.


* * *

##### **C.3 — Budget Constraint & Hybrid Strategy**

Relabeling all ambiguous words with a Large LLM is **cost-prohibitive**. Strategy: use deterministic, syntax-driven rules to resolve a subset for free, sending only the hard cases to the LLM.

**Decision**: The rule-based filter is used **both** to (1) reduce LLM relabeling cost during dataset construction **and** (2) operate as the first stage of the final inference pipeline. This makes the fine-tuned model a **semantic specialist** — it only learns ambiguities that rules provably cannot solve.

**Final Inference Pipeline (for unseen sentences)**:

```
Input → DiaParser (HEAD, DEPREL) + Apertium (readings w/ CG)
      → [Rule-Based Filter] resolves clear-cut cases
      → [Fine-tuned Model] resolves remaining semantic ambiguity
      → Fully disambiguated morphology
```

* * *

##### **C.4 — MUDT-Derived Rule-Based Filter**

**Rationale**: The MUDT treebank improves DEPREL/HEAD to better reflect Uyghur grammar. CG3 (`.rlx`) operates only on **linear surface neighborhoods** and cannot see the dependency tree; therefore rules must be applied in a **custom post-processing script** that aligns Apertium output with the parsed CoNLL-U tree (reusing the existing N:M monotonic alignment script).

**Evaluated three MUDT improvements:**

**Zero-Copula Rules** — Apertium over-generates the null 3rd-person copula `+ئى<cop><aor><p3><sg>`, attaching it to nearly every nominal/numeral/adjective "just in case" it is the predicate. This maps directly onto MUDT's explicit `cop:zero` relation.

*   **Rule 1a — Negative filter**: If a token's DEPREL is a clear modifier/argument (`amod`, `nummod`, `det`, `case`, `nmod`, `obj`, `obl`, `advmod`, `mark`), it cannot be the clause predicate → **delete all `<cop>` readings**.

*   **Rule 1b — Positive filter**: If a token is the **HEAD of a `cop:zero` arc** (strong signal) or a `root` with nominal UPOS (weak signal), it **is** the predicate → **keep only `<cop>` readings** (fires only when a copula candidate exists; safe because Apertium only emits the 3rd-person copula, which matches the bare predicate surface form).


**Linguistic safety note**: 1st/2nd person predicates carry overt agreement suffixes and are analyzed as different surface forms by Apertium, so they never produce the `<p3>` copula candidate — the positive rule cannot misfire on them.

* * *

##### **C.5 — Rule Filter Results (MUDT, full dataset)**

| Metric | Count |
| --- | --- |
| Total sentences | 3,456 |
| Skipped (alignment failure) | 241 |
| **Ambiguous words encountered** | **14,142** |
| ✅ Fully resolved to 1 reading | 2,457 |
| ⚠️ Partially reduced | 3,179 |
| ❌ Untouched (→ LLM) | 8,506 |

**Breakdown by rule:**

| Rule | Fully resolved | Partially reduced |
| --- | --- | --- |
| 1a Negative (modifier → drop `<cop>`) | 2,366 | 3,141 |
| 1b Positive-strong (`cop:zero` head → keep `<cop>`) | 46 | 14 |
| 1b Positive-weak (`root`+nominal → keep `<cop>`) | 45 | 24 |

*   **Rule-based intervention rate**: ~40% of ambiguous words touched.

*   **Observation**: The negative copula filter does the overwhelming bulk of the work. The positive filter contributes modestly but with high precision (strong `cop:zero` signal).


* * *

##### **C.6 — Prompt Engineering**

**Pipeline integration**: OpenRouter API; output stored as **JSONL** (one ambiguous-word instance per line) for direct downstream use with HuggingFace `datasets`.

**Candidate pre-processing — Deduplication**: Apertium sometimes emits linguistically **identical** readings that differ only in internal tag ordering (e.g., `<pl><px3sp>` vs `<px3pl>`), which the gloss map renders as distinct strings. These are collapsed by **normalized feature set** *before* prompting, to avoid forcing the LLM to fabricate a non-existent distinction. If dedup leaves a single reading → label directly (no LLM call).

**Selection principle** (finalized): The LLM selects the candidate whose features are **all** correct for the context. A candidate must be rejected if it contains **even one** incorrect feature, regardless of how many other features are correct. A candidate with **fewer (even a single) features can be correct** — there is **no bias** toward more or fewer features.

**Reasoning schema**: Final response format:

```json
{
  "reasoning": "<ONE sentence: why the chosen candidate's features are all correct>",
  "correct_id": <integer, 1-indexed>
}
```

* * *

##### **C.7 — Output Format for Fine-Tuning**

Each instance stored with **both** a classification label and a serialized generative target, to support all three planned model families:

```json
{
  "sentence": "...",
  "target_word": "...",
  "candidates": ["...", "..."],
  "label_id": 1,
  "reasoning": "...",
  "target_text": "REASON: <sentence> | ANSWER: 2",
  "source": "rule_based | dedup | llm_based"
}
```
---

### 3. Dictionary Lookup

- No publicly available, electronic Uyghur–English dictionary published in the last ~30 years
- **TODO**: Explore alternative approaches to construct a word list or dictionary suitable for this pipeline


---

## Key Resources

| Resource | Description | Link/Path                                                                                        |
|---|---|--------------------------------------------------------------------------------------------------|
| MUDT | Modern Uyghur Dependency Treebank + DiaParser training script | [GitHub](https://github.com/wyqmath/MUDT)                                                        |
| apertium-uig | Morphological analyser + CG tagger for Uyghur | [GitHub](https://github.com/apertium/apertium-uig)                                               |
| UyUDT | Uyghur UD Treebank (CoNLL-U) | [GitHub](https://github.com/UniversalDependencies/UD_Uyghur-UDT)                                 |
| `conllu-morph.py` | FEATS assignment via DEPREL-based scoring (used to generate UyUDT FEATS) | [`ud-scripts/conllu-morph.py`](https://github.com/ftyers/ud-scripts/blob/master/conllu-morph.py) |
| `uig-feats.tsv` | Mapping: Apertium tags → UD features | [`uig-feats.tsv`](apertium-uig/texts/uig-feats.tsv)                                              |
| `uig-feat-rel.tsv` | DEPREL–FEAT co-occurrence frequencies | [`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv)                                        |