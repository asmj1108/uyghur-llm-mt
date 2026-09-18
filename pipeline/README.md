# Uyghur Linguistic Translation Pipeline

An end-to-end neuro-symbolic pipeline that extracts structured linguistic information (syntax, morphology, and dictionary definitions) for Uyghur sentences and formats it into rich in-context prompts to guide Large Language Models (LLMs) to produce accurate, high-fidelity English translations.

---

## Architecture Overview

```
                        Input Sentence (Uyghur)
                                   │
         ┌─────────────────────────┼────────────────────────┐
         │                         │                        │
         ▼                         ▼                        ▼
[1] Dependency Parser   [2] Morphological Tagger  [3] Dictionary Lookup
    (MUDT DiaParser)        (Apertium + Qwen 2B)      (Schwarz SQLite DB)
         │                         │                        │
       HEAD,                     LEMMA,                 Word-Level
      DEPREL                  XPOS, FEATS              Translations
         │                         │                        │
         └─────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                 [4] In-Context Linguistic Prompt Builder
                     (Clause Structure + Markdown Table)
                                   │
                                   ▼
                   [5] Downstream LLM Translation
                           (Uyghur → English)
```

---

## Key Components

### 1. Dependency Parser (`pipeline.parser`)
- **Engine**: MUDT-trained XLM-RoBERTa DiaParser model (`MUDT/exp/mudt_xlmr_base/model`).
- **Functionality**: Predicts dependency trees (`HEAD`, `DEPREL`), identifying clause roots, subjects (`nsubj`, `csubj`), objects (`obj`, `iobj`), modifiers (`amod`, `det`), and zero-copula arcs (`cop:zero`).
- **Predicate Strength**: Evaluates strong predicate signals (head of `cop:zero` arc) and weak predicate signals (`root`).

### 2. Morphological Tagger & Disambiguator (`pipeline.morphology`)
- **Apertium Morphological Analyzer**: Generates candidate word stems and morphosyntactic features.
- **Rule-Based Syntactic Filter**: Applies MUDT zero-copula rules (Rule 1a Negative Modifier Filter and Rule 1b Positive Predicate Filter) to resolve non-semantic ambiguities.
- **Qwen-2B Causal LM Disambiguator**: Fine-tuned model (`disambiguater/final_model_qwen2b/best`) selects contextually accurate analyses for complex morphological ambiguities.
- **Outputs**: Disambiguated `LEMMA`, `POS`, grammatical `FEATS`, and enclitic particles (`<cop>`, `<qst>`, `<postadv>`).

### 3. Uyghur-English Dictionary (`pipeline.dictionary`)
- **Source**: Digitalized Henry G. Schwarz *An Uyghur-English Dictionary* (`dictionary/uig_eng_dict_complete.sqlite`).
- **Lookup Engine**: High-precision stem matching with:
  - Verb stem hyphenation handling (`كەل-` vs `كەل`)
  - Terminal consonant voicing/devoicing normalization (`ب/پ`, `گ/ك`, `غ/ق`, `د/ت`, `ج/چ`, `ز/س`)
  - ULY Latin transliteration fallback
- **Output**: Clean word-level English translations and parts of speech.

### 4. Prompt Builder (`pipeline.prompt`)
- Assembles high-density, structured in-context prompts comprising:
  1. Original Uyghur Source Sentence
  2. Syntactic Clause Architecture (Main Predicate, Core Arguments, Modifiers)
  3. Word-by-Word Morphosyntactic Breakdown Table (ID, Word, Lemma, Morphology, Head, Dictionary Gloss)
  4. Linguistic Translation Guidelines (pro-drop resolution, verbal agreement, inflection preservation)

### 5. LLM Translator (`pipeline.translator`)
- Translates Uyghur sentences using OpenRouter, OpenAI, or local OpenAI-compatible endpoints.

---

## Usage

### 1. Command Line Interface (`run_pipeline.py`)

#### Single Sentence Analysis:
```bash
python run_pipeline.py --sentence "بۇ بىر ياخشى كىتاب ." --show_prompt
```

#### Single Sentence with LLM Translation:
```bash
export OPENROUTER_API_KEY="your-api-key"
python run_pipeline.py --sentence "ئۇ مەكتەپكە باردى ." --translate --show_prompt
```

#### Batch Processing a Text File:
```bash
python run_pipeline.py --input_file input.txt --output_file results.jsonl --batch_size 16
```

#### Interactive REPL Mode:
```bash
python run_pipeline.py --interactive
```

#### Run Pre-configured Demo:
```bash
python run_pipeline.py --demo --show_prompt
```

---

### 2. Python API

```python
from pipeline import UyghurLinguisticPipeline, PipelineConfig

# 1. Initialize pipeline
pipeline = UyghurLinguisticPipeline()

# 2. Process a single sentence
sentence = "بىز تۈنۈگۈن كونا دوستىمىزنى كۆردۇق ."
result = pipeline.process_sentence(sentence)

# Inspect token breakdown
for tok in result.analysis.tokens:
    print(f"[{tok.id}] {tok.form} -> Lemma: {tok.lemma} | Syntax: {tok.deprel} -> #{tok.head_id} | Dict: {tok.dict_gloss}")

# Inspect generated prompt
print(result.prompt)

# 3. Batch processing
batch_sentences = [
    "بۇ بىر ياخشى كىتاب .",
    "ئۇ مەكتەپكە باردى ."
]
results = pipeline.process_batch(batch_sentences)
```


---

🔍 UYGHUR LINGUISTIC PIPELINE ANALYSIS
======================================================================

Source: بىز تۈنۈگۈن كونا دوستىمىزنى كۆردۇق .
  ----------------------------------------------------------------------
TOKEN BREAKDOWN:
[ 1] بىز             | Lemma: بىز          | POS: Pronoun      | Syntax: nsubj    -> #5  | Dict:
biz (pro.): We
[ 2] تۈنۈگۈن         | Lemma: تۈنۈگۈن      | POS: Adverb       | Syntax: advmod   -> #5  | Dict:
tünügün (n.): Yesterday
[ 3] كونا            | Lemma: كونا         | POS: Adjective    | Syntax: amod     -> #4  | Dict:
kona (adj.): Old
[ 4] دوستىمىزنى      | Lemma: دوست         | POS: Noun         | Syntax: obj      -> #5  | Dict:
dost (n.): Friend
[ 5] كۆردۇق          | Lemma: كۆر          | POS: Standard verb | Syntax: root     -> #0  | Dict:
kör- (vt., vi.): To see, read; To blame
[ 6] .               | Lemma: .            | POS: Punctuation  | Syntax: punct    -> #5  | Dict:
-
======================================================================

Generated In-Context Prompt:
You are an expert Uyghur-to-English translator and computational linguist.
Translate the following Uyghur sentence into fluent, accurate English using the provided linguistic
analysis.

### Uyghur Source Sentence:
بىز تۈنۈگۈن كونا دوستىمىزنى كۆردۇق .

### Clause Architecture & Syntactic Skeleton:
  - **Main Predicate / Action**: كۆردۇق [Lemma: كۆر ('To see, read'), Features: Standard verb,
    Transitive, Past definite, First person, Plural]
  - **Subject(s)**: بىز ('We') (nsubj -> #5 كۆردۇق)
  - **Object(s)**: دوستىمىزنى ('Friend') (obj -> #5 كۆردۇق)
  - **Adverbial / Obliques**: تۈنۈگۈن ('Yesterday') (advmod -> #5 كۆردۇق)

### Word-by-Word Morphosyntactic Breakdown & Dictionary Entries:

| ID                | Word                                                                        | Lemma   | Morphological Features                                         | Syntactic Role (Dependency) | Dictionary Translation  |
|-------------------|-----------------------------------------------------------------------------|---------|----------------------------------------------------------------|-----------------------------|-------------------------|
| 1                 | بىز                                                                         | بىز     | Pronoun, Personal, First person, Plural, Nominative            | nsubj -> #5 (كۆردۇق)        | biz                     |
| (pro.): We        |                                                                             |         |                                                                |                             |                         |
| 2                 | تۈنۈگۈن                                                                     | تۈنۈگۈن | Adverb                                                         | advmod -> #5 (كۆردۇق)       | tünügün (n.): Yesterday |
| 3                 | كونا                                                                        | كونا    | Adjective                                                      | amod -> #4 (دوستىمىزنى)     | kona (adj.): Old        |
| 4                 | دوستىمىزنى                                                                  | دوست    | Noun, First person plural possessive, Accusative               | obj -> #5 (كۆردۇق)          |                         |
| dost (n.): Friend |                                                                             |         |                                                                |                             |                         |
| 5                 | كۆردۇق                                                                      | كۆر     | Standard verb, Transitive, Past definite, First person, Plural | root -> #0                  |                         |
| (ROOT)            | kör- (vt., vi.): To see, read; To blame; 3. vt. To regard, look upon, treat |         |                                                                |                             |                         |
| 6                 | .                                                                           | .       | Punctuation                                                    | punct -> #5 (كۆردۇق)        | -                       |

### Translation Guidance:
  1. Preserve all grammatical inflections (tense, mood, voice, polarity, person/number agreements).
  2. Resolve pro-drop (omitted subjects) using verbal agreement suffixes where applicable.
  3. Use the word-level dictionary definitions in proper contextual syntax.
  4. Provide only the final fluent English translation without conversational preamble.

### English Translation:
