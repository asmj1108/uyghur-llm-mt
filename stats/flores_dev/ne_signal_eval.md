# Named-entity signal evaluation — flores dev (`stats/flores_dev/ne_signal_eval.md`)

Measured 2026-09-10 on the post-fix run (997 sentences, 14,203/17,159 content
tokens glossed = 82.8%). "Confirmed" = romanized Uyghur form/lemma (UEY→ULY)
within edit-distance 2 of a capitalized, non-sentence-initial English word in the
aligned flores reference (`eng_Latn.dev`). This pseudo-gold is a **lower bound**:
phonetic renderings (شياڭگاڭدا=Hong Kong, جون=John, ھەسەن=Hassan) never confirm.

## Rules applied to the 2,956 still-missing content tokens

| rule | tokens | confirmed | P (lower bound) |
|---|---|---|---|
| Apertium `<np>` (pos="Proper noun") | 148 | 66 | 0.45 |
| `<np>` + name subtag (Toponym/Anthroponym/Organisation/Cognomen/Patronymic) | 148 | 66 | 0.45 |
| Latin-script token | 133 | 106 | 0.80 |
| np ∪ name-subtag ∪ Latin | 281 | 172 | 0.61 |

## Conclusions

- **No parser UPOS complement exists**: DiaParser's MUDT checkpoint has no POS
  tagger — `upos` is `_` for every token (verified against model output
  `sent.values`). A parser-`PROPN` NE signal would require retraining the parser
  with a POS head (out of scope).
- The "false positives" of the np rule (unconfirmed tokens) are almost all true
  NEs the transliteration gold cannot match; expected true precision is higher
  than 0.45–0.61.
- **Most reliable NE identification remains eval-time**: the parallel English
  reference via transliteration (report category `en-ne`, 364 tokens) is precise
  by construction.
- **Safe inference-time use**: annotating Apertium `<np>`+name-subtag or
  Latin-script tokens as "proper noun (no gloss needed)" in the prompt never
  fabricates a gloss — it only adds a label, so the low precision is low-risk.