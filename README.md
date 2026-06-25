# Thesis Progress — Memory Protocol

**Topic**: Improving machine translation (Uyghur → English) for a low-resource language via in-context learning with
LLMs

---

## Goal

Build a pipeline that extracts linguistic information for an input sentence and presents it as structured context in a
prompt to guide an LLM to produce better translations.

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

**Tool**: [`apertium-uig`](https://github.com/apertium/apertium-uig) — morphological analyser/generator + Constraint
Grammar (CG) tagger for Uyghur

**Problem**: The `uig-tagger` mode uses CG for disambiguation, but the grammar rules are sparse → some tokens retain *
*multiple readings** after CG

#### Data available for disambiguation

[UyUDT](https://github.com/UniversalDependencies/UD_Uyghur-UDT) — Uyghur UD Treebank in CoNLL-U format:

| Field              | Source                                                                                                                                 |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| UPOS, HEAD, DEPREL | Manual annotation                                                                                                                      |
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

- **Action**: Wrote an N:M monotonic alignment script to map Apertium output to CoNLL-U tokens (achieved >99% pipeline
  alignment compatibility).
- **Result**: Tested the UPOS tag filter hypothesis across the entire UyUDT dataset (16,866 ambiguous words).
    - Ambiguity Reduced: Only ~25%
    - Still ambiguous: ~74%
    - Word lost entirely (Overfiltered): ~12%
- **Conclusion**: UPOS filtering is definitively insufficient for Uyghur morphological disambiguation. UPOS is too
  coarse to resolve internal morphological ambiguity (e.g., Noun vs. Noun readings), and pipeline tokenization conflicts
  lead to unacceptably high data loss (12%).
- **Decision**: Abandoning standalone UPOS filtering.

#### Approach B: Disambiguate by DEPREL (used in UyUDT, for reference)

**Method** (see [`ud-scripts/conllu-morph.py`](ud-scripts/conllu-morph.py) + [
`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv)):

- Score each candidate reading by:
    1. Overlap between candidate tags and existing annotation
    2. Co-occurrence frequency of the token's DEPREL with each FEAT value (from `uig-feat-rel.tsv`)
- Select highest-scoring reading

**Limitations**:

- Statistical heuristic — origin of frequency counts in [`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv) is
  unknown
- Not linguistically/grammatically motivated
- Sub-optimal as a general solution

#### Approach C: Disambiguate using LLM

**Concept**: Large LLMs have deep knowledge of Universal Dependencies from multilingual pre-training. Use a Large LLM to
generate high-quality **"Pseudo-Gold"** labels by selecting the correct reading for each ambiguous word, then fine-tune
a smaller model for fast, production-ready pipeline inference.

**Architecture**: Neuro-Symbolic Hybrid Pipeline — deterministic syntactic rules resolve what they can with 100%
confidence; the LLM (and later, the fine-tuned model) handles only the semantic residual.

* * *

#### **C.1 — Tag-to-Gloss Formatting**

- Wrote a Python script utilizing `streamparser` to intercept complex Apertium readings (e.g., `<v><tv><ger><nom>`) and
  transform them into human-readable UD-style glosses via a dictionary mapping scheme based on the documentation of
  apertium symbols.
- Handles `+`-joined lexical units (e.g., the enclitic copula `+ئى<cop>`) by splitting and independently glossing each
  morphological component.
- **Output**: Structured, text-based candidate descriptions ready for LLM prompting and classification.
- **Label hygiene** (post-hoc audit of the generated label space `label_vocab.json`):
    - **Unmapped tags fixed**: Several Apertium symbols (`che`, `ter`, `iver`, `td`, `ger_past2`, `gna_cond`,`gpr_fut2`,
      `past3`) were passing through the fallback as raw uppercase strings
    - **`ERR_ORTH` stripped**: This is a meta-annotation for orthographic variants, not a grammatical feature → removed
      before feature extraction so it never enters a candidate.
    - **Variant-number fallback**: The variant digit of features is striped (e.g., gpr_fut2 → gpr_fut),
      collapsing spurious numbered variants onto their base gloss.
    - **Malformed-reading guard**: Fixed a data-corruption case where a surface form leaked into a tag slot when a
      reading is dropped entirely if its main lexical unit has no lemma.

#### **C.2 — Key Discovery: Human Annotation Inconsistencies in UyUDT**

Testing revealed instances where manual human annotations in UyUDT deviate from strict UD syntactic principles.

* **Example**: The token `تۇغقاننىڭ` ("relative's") in sentence `s780` was manually assigned a UPOS of `ADJ`. This label
  was potentially influenced by the presence of the participial suffix `-قان`, despite the word functioning structurally
  as a fully lexicalized `NOUN` in this specific context. In contrast, the programmatic XPOS (derived from Apertium)
  analyzed it as `N`. During initial testing, the LLM correctly identified the noun reading.
* **Implication**: Human manual annotations are not an infallible gold standard. While Large LLMs are certainly capable
  of making errors, current testing suggests they possess strong contextual reasoning abilities and can likely produce
  labels that are **superior to the available human baseline** for historically tricky morpho-syntactic cases.

#### **C.3 — The Fallacy of Filtering by UPOS/XPOS Mismatches**

Given the LLM API budget constraints (16,866 ambiguous tokens requires filtering), an intuitive strategy was proposed:
only relabel tokens where the human `UPOS` tag disagrees with the Apertium-derived `XPOS` tag. **This approach was
abandoned due to two critical methodological flaws:**

1. **The Matching Tags Problem (Feature-Level Ambiguity)**: An Apertium module frequently outputs multiple readings with
   the *same* POS but different features (e.g., Candidate A: Noun/Nominative vs. Candidate B: Noun/Accusative).
   CoNLL-U's `UPOS` and `XPOS` will naturally match (`NOUN` == `N`), falsely signaling "no mismatch." However, the word
   remains highly ambiguous; selecting the wrong case will cause downstream dependency parsing to fail.
2. **The Mismatched Tags Problem**: Mismatches can be caused by both erroneous human UPOS tags (as demonstrated in C.2)
   and erroneous apertium-assigned XPOS tags (see the suboptimal method in Approach B) rather than actual morphological
   complexity. Using POS tags to decide what needs "fixing" inherits the errors.

* * *

#### **C.4 — Budget Constraint & Hybrid Strategy**

Relabeling all ambiguous words with a Large LLM is **cost-prohibitive**. Strategy: use deterministic, syntax-driven
rules to resolve a subset for free, sending only the hard cases to the LLM.

**Decision**: The rule-based filter is used **both** to (1) reduce LLM relabeling cost during dataset construction **and
** (2) operate as the first stage of the final inference pipeline.

**Final Inference Pipeline (for unseen sentences)**:

```
Input → DiaParser (HEAD, DEPREL) + Apertium (readings w/ CG)
      → [Rule-Based Filter] resolves clear-cut cases
      → [Fine-tuned Model] resolves remaining semantic ambiguity
      → Fully disambiguated morphology
```

#### **C.5 - MUDT-Derived Rule-Based Filter**

**Rationale**: The MUDT treebank improves DEPREL/HEAD to better reflect Uyghur grammar. CG3 (`.rlx`) operates only on *
*linear surface neighborhoods** and cannot see the dependency tree; therefore rules must be applied in a **custom
post-processing script** that aligns Apertium output with the parsed CoNLL-U tree (reusing the existing N:M monotonic
alignment script).

**Zero-Copula Rules** — Apertium over-generates the null 3rd-person copula `+ئى<cop><aor><p3><sg>`, attaching it to
nearly every nominal/numeral/adjective "just in case" it is the predicate. This maps directly onto MUDT's explicit
`cop:zero` relation.

* **Rule 1a — Negative filter**: If a token's DEPREL is a clear non-predicate — modifier, core argument, or **subject
  ** (`amod`, `nummod`, `det`, `case`, `nmod`, `obj`, `obl`, `advmod`, `mark`, `nsubj`, `csubj`, `iobj`) — it cannot be
  the clause predicate → **delete all `<cop>` readings**.

    * **On including `nsubj`/`csubj`**: This is principled under MUDT's own design — MUDT routes the subject of a
      *nominal* predicate through `cop:zero`, reserving `nsubj`/`csubj` for subjects of *verbal* predicates. A subject
      is categorically never its own predicate, so it cannot host the null copula and is safe to negative-filter. (
      Confirmed on `s2211`: `كۆزلىرى` is `nsubj` of a verbal predicate, with spurious copula readings correctly
      dropped.)

* **Rule 1b — Positive filter**: If a token is the **HEAD of a `cop:zero` arc** (strong signal) or a **`root`** (weak
  signal), it **is** the predicate → **keep only `<cop>` readings** (fires only when a copula candidate exists).

**Determining the nominal predicate without UPOS**: At inference we have only DiaParser output (HEAD, DEPREL) + Apertium
readings — **no UPOS**. Since the filter must consume identical inputs at build and inference time, nominality is
determined via a proxy already available in both: Apertium emits the null 3rd-person copula candidate **only on bare
nominal/adjectival/numeral predicates** (a finite verbal `root` has a different surface form and produces no copula
reading), so the mere presence of a copula candidate guarantees nominality, making an explicit UPOS test redundant. The
entire filter therefore depends solely on **{HEAD, DEPREL} + Apertium readings**, achieving train/inference parity.

**Linguistic safety note**: 1st/2nd person predicates carry overt agreement suffixes and are analyzed as different
surface forms by Apertium, so they never produce the `<p3>` copula candidate — the positive rule cannot misfire on them.

**Limitation (noted)**: A residual train/inference gap remains — the builder uses gold MUDT trees while inference uses
predicted DiaParser trees, so parser errors can flip rule decisions; this is the standard distillation gap and is
accepted.

##### **Rule Filter Results (full dataset)**

| Metric                          | Count     |
|---------------------------------|-----------|
| Total sentences                 | 3,456     |
| Skipped (alignment failure)     | 241       |
| **Ambiguous words encountered** | **15438** |
| Fully resolved to 1 reading     | 4259      |
| Partially reduced               | 4265      |
| Untouched (→ LLM)               | 6914      |

**Breakdown by rule:**

| Rule               | Fully resolved | Partially reduced |
|--------------------|----------------|-------------------|
| 1a Negative        | 3,718          | 4,118             |
| 1b Positive-strong | 66             | 20                |
| 1b Positive-weak   | 475            | 127               |

* **Rule-based intervention rate**: ~55% of ambiguous words touched.

* * *

#### **C.6 — Dataset Builder (assembled pipeline)**

The dataset-builder integrates all components into the agreed cascade, writing JSONL for direct HuggingFace datasets
use, with checkpoint/resume and an async LLM phase (OpenRouter).

**Candidate pre-processing — Deduplication**: Apertium emits readings that are indistinguishable to our feature-based
representation — either pure tag-ordering variants or readings differing only in the target-word lemma. These are
collapsed by normalized feature set before prompting. If dedup leaves a single reading → label directly (source="dedup",
no LLM call).

**Selection principle**: The LLM selects the candidate whose features are **all** correct for the context. A
candidate must be rejected if it contains **even one** incorrect feature, regardless of how many other features are
correct. A candidate with **fewer (even a single) features can be correct** — there is **no bias** toward more or fewer
features.

**LLM response format**:

```json
{
  "reasoning": "<ONE sentence: why the chosen candidate's features are all correct>",
  "correct_id": <integer, 1-indexed>
}
```


**Output schema** — every row carries both a classification target and a serialized generative target, plus
structured fields for the encoder:

| Field                            | Purpose                                                                                                            |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `sentence`, `target_word`        | context + focus                                                                                                    |
| `candidates`                     | natural-language glosses (lemma included) → **decoder** input                                                      |
| `candidate_feats`, `label_feats` | canonical feature-label lists (target lemma omitted; enclitics kept as one atomic label) → **encoder** multi-label |
| `label_id`                       | gold index into candidates                                                                                         |
| `reasoning`                      | rationale (LLM-generated, or deterministic for rule/dedup rows)                                                    |
| `target_text`                    | `"REASON: <rationale>`                                                                                             | ANSWER: <id>"` → **decoder** target |
| `source`                         | `dedup` \| `rule_based` \| `llm_based` (enables filtering to pure distillation)                                    |

* * *

#### **C.7 — Finetuning a Encoder Model**
**STRUCTURE**
```

**XLM-RoBERTa**

```aiignore
                        INPUT ROW (one ambiguous word)
   sentence: "... دوست بولۇپ ..."   target_word: "دوست"
   label_feats:      [Noun, Nominative]              ← gold answer
   candidate_feats:  [[Noun,Nominative],             ← candidate 0
                      [Noun,Nominative,Copula,...]]  ← candidate 1
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │ STEP A — Target grounding                                  │
   │   Wrap target word in special tokens so the model knows    │
   │   WHICH word to analyse:                                   │
   │   "... <t> دوست </t> بولۇپ ..."                            │
   └──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │ STEP B — Encode + multi-label head                         │
   │   XLM-RoBERTa  →  one sigmoid probability per feature      │
   │   label        →  multi-hot vector from label_feats        │
   │   loss         →  focal BCE (handles label sparsity)       │
   └──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │ STEP C — Decode back to a CANDIDATE (eval only)            │
   │   We DON'T threshold at 0.5. Instead, for each candidate   │
   │   feature-set we compute its log-likelihood under the      │
   │   predicted probabilities, and pick the best-scoring one.  │
   │   → compare to gold label_id  →  cand_accuracy             │
   └──────────────────────────────────────────────────────────┘
```
**RESULT**
```aiignore
                  CANDIDATE ACCURACY  (higher = better)
   0.0       0.2       0.4       0.6       0.8       1.0
   |---------|---------|---------|---------|---------|
   random          ████████████████ 0.41
   most-features   █████ 0.13                ← worst: "more = better" is WRONG
   fewest-features ████████████████████████████████ 0.78
   first-candidate ██████████████████████████████████ 0.81
   ★ OUR MODEL     ████████████████████████████████████▌ 0.876   (test)
```
#### **C.8 — Finetuning a Seq2Seq Model**

ByT5

📊 BASELINES on test.jsonl  (n=1404)

first           : 0.8077
random          : 0.3996
most-features   : 0.1147
fewest-features : 0.7721

---

### 3. Dictionary Lookup

- No publicly available, electronic Uyghur–English dictionary published in the last ~30 years
- **TODO**: Explore alternative approaches to construct a word list or dictionary suitable for this pipeline

---

## Key Resources

| Resource           | Description                                                              | Link/Path                                                                                        |
|--------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| MUDT               | Modern Uyghur Dependency Treebank + DiaParser training script            | [GitHub](https://github.com/wyqmath/MUDT)                                                        |
| apertium-uig       | Morphological analyser + CG tagger for Uyghur                            | [GitHub](https://github.com/apertium/apertium-uig)                                               |
| UyUDT              | Uyghur UD Treebank (CoNLL-U)                                             | [GitHub](https://github.com/UniversalDependencies/UD_Uyghur-UDT)                                 |
| `conllu-morph.py`  | FEATS assignment via DEPREL-based scoring (used to generate UyUDT FEATS) | [`ud-scripts/conllu-morph.py`](https://github.com/ftyers/ud-scripts/blob/master/conllu-morph.py) |
| `uig-feats.tsv`    | Mapping: Apertium tags → UD features                                     | [`uig-feats.tsv`](apertium-uig/texts/uig-feats.tsv)                                              |
| `uig-feat-rel.tsv` | DEPREL–FEAT co-occurrence frequencies                                    | [`uig-feat-rel.tsv`](apertium-uig/texts/uig-feat-rel.tsv)                                        |