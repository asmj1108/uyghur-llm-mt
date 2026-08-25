# PARSING_SPECIFICATION.md - Uyghur-English Dictionary Lexicographical Parser Specification

This document provides a comprehensive technical specification of the parsing logic implemented in `parser.py` for Henry G. Schwarz's *An Uyghur-English Dictionary* (1992).

---

## 1. Overview & Objectives

The parser transforms unstructured, markdown-formatted column transcriptions produced by the OCR/multimodal transcription stage into a rich, fully normalized lexicographical data structure.

### Input Data Format
- Headwords formatted in markdown bold (`**word**`, `**word I**`, `**word** I`).
- Numbered senses formatted as `1.`, `2.`, `3.`.
- Uyghur phrases, idioms, sayings, and examples italicized with single asterisks (`*...*`).
- Placeholders `--`, `—`, and `–` representing the headword or its inflected stem.
- Special meanings marked with diamonds (`♦`).
- Sayings and proverbs marked with pilcrows (`¶`).
- Loanword etymologies and Turkic cognates enclosed in square brackets (`[...]`).
- Cross-references indicated by `Also`, `Also called`, or `Cf.`.
- Column continuations prefixed with `[CONTINUATION]`.

### Output Target
- High-fidelity JSON dataset matching the canonical digital schema.
- Normalized SQLite relational database with an FTS5 full-text index across Uyghur Arabic (UEY), standard Latin (ULY), Schwarz Latin, and English definitions.

---

## 2. Canonical JSON Schema

Each dictionary entry produces a structured JSON object with the following schema:

```json
{
  "headword": {
    "schwarz": "xam",
    "uly": "xam",
    "uey": "خام",
    "uyy": "ham",
    "ipa": "χɑm"
  },
  "homograph_number": "I",
  "pos": ["adj."],
  "domain": [],
  "page_book": 374,
  "page_pdf": 399,
  "column": 2,
  "senses": [
    {
      "sense_number": 1,
      "pos": "adjective",
      "voice": null,
      "domain": null,
      "definition": "Crude; fresh; raw",
      "examples": [
        {
          "uyghur_schwarz": "xam terä",
          "uyghur_uey": "خام تېرە",
          "uyghur_uly": "xam tëre",
          "domain": null,
          "english": "raw leather"
        }
      ],
      "sayings": [],
      "special_meanings": [
        {
          "uyghur_schwarz": "xam süt ämgän bändä",
          "uyghur_uey": "خام سۈت ئەمگەن بەندە",
          "uyghur_uly": "xam süt emgen bende",
          "english": "sby drinking raw milk (a greenhorn; sby likely to make a mistake)."
        }
      ],
      "cross_references": [],
      "etymologies": [],
      "correspondences": [],
      "raw_text": "adj. Crude; fresh; raw: *-- terä* raw leather/ ... ♦ *-- süt ämgän bändä* ..."
    }
  ],
  "raw_body": "1. adj. Crude; fresh; raw: ... 2. adj. New; unfamiliar: ... 3. adj. Inexperienced: ..."
}
```

---

## 3. Lexicographical Parsing Architecture

The parsing pipeline in `parser.py` operates through distinct modular stages:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Column Transcription Tokenizer                           │
│    - Detects bold headwords & homographs                    │
│    - Extracts leading [CONTINUATION] text                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 2. Cross-Column / Multi-Page Stitcher                       │
│    - Reconnects hyphenated word breaks across columns/pages │
│    - Resolves orphan headwords with bodies on next page     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 3. Sense Segmentation & Sequential Numbering                │
│    - Splits strictly on sequential integers (1., 2., 3., ...)│
│    - Avoids false splits on dates (1983.) or homographs     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 4. Sense Component Extraction                               │
│    - Bracketed info: Etymology (<) vs Correspondence        │
│    - Cross-references: Also, Also called, Cf.               │
│    - Proverbs (¶) with parenthetical explanations           │
│    - Special meanings (♦) with multi-expression colons      │
│    - POS, Voice, Subject Domain tags                        │
│    - Definition references converted to UEY                 │
│    - Uyghur example phrases (/) separated from English      │
│    - Headword re-embedding (-- / — / –)                     │
│    - 4-script transliteration + IPA generation              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Parsing Rules & Algorithms

### 4.1. Headword & Homograph Extraction
- **Headword Pattern**:
  ```python
  re.match(r'^\*\*[^*]+(?:\*\*(?:\s+[IVXLCDM\d]+)?|\s+[IVXLCDM\d]+\*\*)\s*$', line.strip())
  ```
- **Homograph Recognition**:
  Extracts Roman numerals (`I`, `II`, `III`, `IV`, etc.) or numbers whether formatted inside bold asterisks (`**xam I**`) or outside (`**xam** I`).
- **Headword Normalization**:
  Removes markdown formatting and normalizes Schwarz diacritics (`g̃`, `ñ`, `ⱬ`, `ä`, `ö`, `ü`, `ç`, `ş`).

### 4.2. Multipage & Column Continuity
- **Word-Wrap Dehyphenation**:
  When an entry body ends with a word-split hyphen (e.g. `aris-`) and the continuation begins with a lowercase letter (e.g. `tocratic`), `stitch_body_text` joins them into `aristocratic`.
- **Orphan Headword Resolution**:
  When a headword appears at the very bottom of a column or page with an empty body (e.g. `xaniwäyrançiliq` on page 400), and the next column starts with that body labeled by a hallucinated running header (e.g. `xaniwäyranliq`), the stitching logic recognizes the empty body and assigns the continuation text to the orphan headword, removing the hallucinated headword.

### 4.3. Numbered Sense Splitting
- **Sequential Splitting**:
  `split_into_numbered_senses` checks for leading `1.` and strictly searches for the next sequential number `2. `, `3. `, etc.
- **Prefix Preservation**:
  Leading compound qualifiers before `1.` (e.g. `abstract n. 1. ...` or `vt. 1. ...`) are preserved and assigned to Sense 1.
- **Date & Citation Safety**:
  Arbitrary numbers like `1983.` or Roman numeral homographs in cross-references are not misidentified as senses because strictly increasing sequence numbers are required.

### 4.4. Headword Placeholder Re-Embedding (`--`, `—`, `–`)
- **Standalone Placeholders**:
  - For nouns/adjectives: `--` → `headword`
  - For verbs (`stem-`): `--` → `stem-`
- **Inflected Suffixes**:
  - `*--lär*` → `headwordlär`
  - `*--i*` → `headwordi`
  - `*--ni*` → `headwordni`
  - `*--qa*` / `*--tin*` with stem phonology (e.g. `at` + `--tin` → `attin`, `at` + `--qa` → `atqa`).
  - Verb converbs: `--p` attaches harmonic linking vowel (`u` or `i`) if stem ends with consonant (e.g. `al-` + `--p` → `elip`, `sat-` + `--p` → `setip`).
  - Negative verb stems: `--mi-` → `stemmi-`.

### 4.5. Special Meanings (`♦`)
- **Marker**: Diamonds (`♦`) denote idioms, fixed expressions, and specialized sub-terms.
- **Sub-Expression Splitting**:
  When a single `♦` block contains multiple related expressions connected by a colon (e.g. `♦ *-- xiyal* fantasy; illusion: -- *xiyal qil-* to have illusions.`), the parser splits them on `: ` followed by Uyghur markers (`--`, `—`, `–`, or `*...*`):
  1. Expression 1: `xam xiyal` → `fantasy; illusion`
  2. Expression 2: `xam xiyal qil-` → `to have illusions.`
- **Descriptive Prefixes**:
  If a fragment before a colon is a descriptive English gloss (e.g. `to blemish: *şänigä -- täg-* to blemish a reputation`), it is preserved as an English prefix.

### 4.6. Proverbs & Sayings (`¶`)
- **Marker**: Pilcrows (`¶`) denote folk proverbs and sayings.
- **Parenthetical Explanations**:
  Metaphorical/cultural explanations inside parentheses at the end of a saying are extracted into the `explanation` field:
  - Text: `¶ *attin çüşsä, üzäñgidin çüşmäydu* getting off a horse but not out of stirrups (expresses idea of not admitting defeat or of not giving up).`
  - `english`: `"getting off a horse but not out of stirrups"`
  - `explanation`: `"expresses idea of not admitting defeat or of not giving up"`

### 4.7. Example Phrases (`/`)
- **Delimiter**: Slash (`/`) separates distinct example phrases under a definition.
- **Uyghur vs. English Separation**:
  Uses single-asterisk boundary matching (`(?<!\*)\*([^*]+)\*(?!\*)`) to isolate the italicized Uyghur text from the English translation.
- **Domain Tag Stripping**:
  Extracts embedded subject domain tags (e.g. `*öt xaltisi* med. gall bladder` → domain `med.`, English `gall bladder`).

### 4.8. Definition Reference Conversion to UEY
Cross-references and derivations within definitions are converted to Uyghur Arabic Script (UEY):
1. **Headword Equivalence**:
   `= xalisanä.` → `= خالىسانە.`
   `= xaniwäyrançiliq.` → `= خانىۋەيرانچىلىق.`
2. **Grammatical Derivations**:
   `abstract n. of xaniwäyran` → `abstract n. of خانىۋەيران`
   `vi. caus. of at-` → `vi. caus. of ئات-`
   `Alternate name of wensu` → `Alternate name of ۋېنسۇ`
3. **Apostrophe / Syllable Divider Support**:
   Correctly captures stems containing glottal dividers (e.g. `= xa'inanä` → `= خائىنانە`).

### 4.9. Etymologies vs. Correspondences
- **Loanword Etymology Chains (`<`)**:
  - Format: `[<R. abonement <F. abonne-]`
  - Parsed into ordered `chain` arrays with donor language code and loanword word.
- **Turkic / Mongolic Correspondences**:
  - Format: `[Ç. at/K. at/U. ot (#1-3)]`
  - Separated by `/`, mapping 14 language abbreviations (`Ç`, `K`, `KK`, `M`, `Mo`, `O`, `OT`, `Q`, `R`, `Ta`, `TT`, `Tu`, `Tv`, `U`, `Y`) to full names with attached sense notes.

### 4.10. Cross-References (`Also`, `Also called`, `Cf.`)
- Identifies cross-reference markers:
  - `Also <target>`
  - `Also called <target>`
  - `Cf. <target>`
- Splits multiple targets separated by commas.
- Normalizes Schwarz Latin targets and transliterates to UEY and standard ULY.

---

## 5. Tag Reference Tables

### 5.1. Parts of Speech (`POS_TAGS`)
| Tag | Full Meaning |
|---|---|
| `adj.` | adjective |
| `adv.` | adverb |
| `conj.` | conjunction |
| `int.` | interjection |
| `m.` | measure word |
| `num.` | numeral |
| `n.` | noun |
| `onom.` | onomatepia |
| `pp.` | postposition |
| `prep.` | preposition |
| `pro.` | pronoun |
| `vi.` | intransitive verb |
| `vt.` | transitive verb |

### 5.2. Grammatical Voice (`VOICE_TAGS`)
| Tag | Full Meaning |
|---|---|
| `caus.` | causative voice |
| `mut.` | mutual voice |
| `pass.` | passive voice |
| `retro.` | retroactive voice |

### 5.3. Subject / Restrictive Domains (`DOMAIN_TAGS`)
| Tag | Full Meaning | Tag | Full Meaning |
|---|---|---|---|
| `ag.` | agriculture | `math.` | mathematics |
| `ast.` | astronomy | `med.` | medicine |
| `bot.` | botany | `mil.` | military |
| `chem.` | chemistry | `mus.` | music |
| `clo.` | clothing | `obs.` | obsolete |
| `cul.` | culinary | `pe.` | physical education; sports |
| `fig.` | figurative | `ph.` | philosophy |
| `geo.` | geography | `phy.` | physics |
| `geol.` | geology | `phys.` | physiology |
| `gr.` | grammar; linguistics | `poe.` | poetic |
| `his.` | historical; history | `rel.` | religion |
| `lit.` | literature and arts | `zoo.` | zoology |

### 5.4. Donor & Cognate Language Codes (`LANG_NAMES`)
| Code | Language | Code | Language |
|---|---|---|---|
| `Ar` | Arabic | `OT` | Orkhon Turkic |
| `Pers` | Persian | `O` | Osmanli Turkish |
| `Ç` | Chagatay | `Q` | Kazakh |
| `Ch` / `Chin` | Chinese | `R` | Russian |
| `E` | English | `Ta` | Tatar |
| `F` | French | `TT` | Turfan texts |
| `G` | German | `Tu` | Turkmen |
| `K` | Kirghiz | `Tv` | Tuvan |
| `KK` | Karakalpak | `U` | Uzbek |
| `M` | Mahmud al-Kaşğari | `Y` | Yakut |
| `Mo` | Mongolian | | |

---

## 6. Database Schema & FTS5 Indexing

The parsed entries map directly into the relational SQLite database (`uyghur_english_dictionary.sqlite`):

```sql
-- Entries table
CREATE TABLE entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headword_schwarz TEXT NOT NULL,
    headword_uey TEXT NOT NULL,
    headword_uly TEXT NOT NULL,
    headword_uyy TEXT NOT NULL,
    headword_ipa TEXT NOT NULL,
    homograph_number TEXT,
    page_book INTEGER,
    page_pdf INTEGER,
    column INTEGER,
    raw_body TEXT
);

-- Senses table
CREATE TABLE senses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER REFERENCES entries(id),
    sense_number INTEGER,
    pos TEXT,
    voice TEXT,
    domain TEXT,
    definition TEXT,
    raw_text TEXT
);

-- Examples table
CREATE TABLE examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    uyghur_schwarz TEXT,
    uyghur_uey TEXT,
    uyghur_uly TEXT,
    domain TEXT,
    english TEXT
);

-- Sayings table
CREATE TABLE sayings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    uyghur_schwarz TEXT,
    uyghur_uey TEXT,
    uyghur_uly TEXT,
    english TEXT,
    explanation TEXT
);

-- Special meanings table
CREATE TABLE special_meanings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    uyghur_schwarz TEXT,
    uyghur_uey TEXT,
    uyghur_uly TEXT,
    english TEXT
);

-- Cross references table
CREATE TABLE cross_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    type TEXT,
    target_schwarz TEXT,
    target_uey TEXT,
    target_uly TEXT
);

-- Etymologies table
CREATE TABLE etymologies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    raw TEXT,
    donor_language TEXT,
    donor_word TEXT
);

-- Correspondences table
CREATE TABLE correspondences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id INTEGER REFERENCES senses(id),
    language TEXT,
    lang_code TEXT,
    word TEXT,
    note TEXT
);

-- Full-text Search FTS5 Virtual Table
CREATE VIRTUAL TABLE entries_fts USING fts5(
    headword_schwarz,
    headword_uey,
    headword_uly,
    definitions,
    examples,
    content='entries',
    content_rowid='id'
);
```
