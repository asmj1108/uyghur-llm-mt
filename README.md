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

---

### 3. Dictionary Lookup

- No publicly available, electronic Uyghur–English dictionary published in the last ~50 years
- **TODO**: Explore alternative approaches to construct a word list or dictionary suitable for this pipeline


---

## Next Steps

- [ ] Explore dictionary alternatives (scraped resources, bilingual corpora, other lexicons, LLM-generated entries, etc.)

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