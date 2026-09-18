"""
prompt.py - In-context prompt builder that presents structured linguistic annotations to LLMs
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from .parser import ParsedSentence, ParsedToken
from .morphology import MorphToken
from .dictionary import DictEntry, uey_to_uly


@dataclass
class TokenAnnotation:
    """Consolidated linguistic annotation for a single token."""
    id: int
    form: str
    lemma: str
    pos: str
    features: List[str]
    enclitics: List[str]
    head_id: int
    head_form: str
    deprel: str
    pred_strength: Optional[str]
    dict_entry: Optional[DictEntry] = None
    near_entries: List[DictEntry] = field(default_factory=list)
    uly: str = ""

    @property
    def dict_gloss(self) -> str:
        if self.dict_entry:
            return self.dict_entry.short_gloss
        if self.pos == "Punctuation":
            return "-"
        if self.near_entries:
            return "; ".join("≈ " + e.short_gloss for e in self.near_entries[:2])
        return f"[{self.lemma}]"

    @property
    def morph_summary(self) -> str:
        if self.pos == "Punctuation":
            return "Punctuation"
        feats_clean = [f for f in self.features if not f.startswith("Lemma:")]
        base = ", ".join(feats_clean) if feats_clean else self.pos
        if self.enclitics:
            encl_str = " + ".join(self.enclitics)
            return f"{base} ({encl_str})"
        return base


@dataclass
class LinguisticAnalysis:
    """Full linguistic analysis for a sentence."""
    sentence: str
    tokens: List[TokenAnnotation] = field(default_factory=list)
    clause_structure: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "clause_structure": self.clause_structure,
            "tokens": [
                {
                    "id": t.id,
                    "form": t.form,
                    "lemma": t.lemma,
                    "uly": t.uly,
                    "pos": t.pos,
                    "morphology": t.morph_summary,
                    "syntax": f"{t.deprel} -> {t.head_form} (#{t.head_id})",
                    "dictionary": t.dict_gloss
                }
                for t in self.tokens
            ]
        }


class PromptBuilder:
    """
    Constructs rich in-context linguistic prompts to guide LLMs in Uyghur -> English translation.
    """

    @staticmethod
    def build_analysis(
        parsed_sent: ParsedSentence,
        morph_tokens: List[MorphToken],
        dict_entries: List[Optional[DictEntry]],
        near_entries: Optional[List[List[DictEntry]]] = None
    ) -> LinguisticAnalysis:
        """Merge parser, morphology, and dictionary outputs into a cohesive analysis."""
        token_annotations: List[TokenAnnotation] = []

        near_entries = near_entries or [[] for _ in dict_entries]

        # Map token ID -> form for head lookups
        id_to_form = {t.id: t.form for t in parsed_sent.tokens}
        id_to_form[0] = "ROOT"

        for i, (p_tok, m_tok, d_entry) in enumerate(zip(parsed_sent.tokens, morph_tokens, dict_entries)):
            head_form = id_to_form.get(p_tok.head, "ROOT")
            token_annotations.append(
                TokenAnnotation(
                    id=p_tok.id,
                    form=p_tok.form,
                    lemma=m_tok.lemma,
                    uly=uey_to_uly(m_tok.lemma) or uey_to_uly(m_tok.form),
                    pos=m_tok.pos,
                    features=m_tok.features,
                    enclitics=m_tok.enclitics,
                    head_id=p_tok.head,
                    head_form=head_form,
                    deprel=p_tok.deprel,
                    pred_strength=p_tok.pred_strength,
                    dict_entry=d_entry,
                    near_entries=near_entries[i] if i < len(near_entries) else []
                )
            )

        # Build high-level clause structure
        clause_structure = PromptBuilder._extract_clause_structure(token_annotations)

        return LinguisticAnalysis(
            sentence=parsed_sent.text,
            tokens=token_annotations,
            clause_structure=clause_structure
        )

    @staticmethod
    def _extract_clause_structure(tokens: List[TokenAnnotation]) -> Dict[str, Any]:
        """Extract main syntactic arguments and roles in the clause."""
        root_tokens = [t for t in tokens if t.deprel == "root" or t.head_id == 0]
        subjects = [t for t in tokens if "subj" in t.deprel]
        objects = [t for t in tokens if t.deprel in ("obj", "iobj")]
        obliques = [t for t in tokens if t.deprel in ("obl", "advmod", "case", "case:dat")]
        modifiers = [t for t in tokens if t.deprel in ("amod", "det", "nummod", "nmod")]

        root_summary = []
        for r in root_tokens:
            dict_str = f" ('{r.dict_entry.definitions[0]}')" if r.dict_entry and r.dict_entry.definitions else ""
            root_summary.append(f"{r.form} [Lemma: {r.lemma}{dict_str}, Features: {r.morph_summary}]")

        def format_tok_list(tok_list):
            items = []
            for t in tok_list:
                dict_str = f" ('{t.dict_entry.definitions[0]}')" if t.dict_entry and t.dict_entry.definitions else ""
                items.append(f"{t.form}{dict_str} ({t.deprel} -> #{t.head_id} {t.head_form})")
            return items

        return {
            "root_predicate": root_summary,
            "subjects": format_tok_list(subjects),
            "objects": format_tok_list(objects),
            "obliques_and_adverbials": format_tok_list(obliques),
            "modifiers": format_tok_list(modifiers)
        }

    @classmethod
    def build_prompt(
        cls,
        analysis: LinguisticAnalysis,
        format_type: str = "markdown_table"
    ) -> str:
        """Render the complete in-context prompt for an LLM."""
        if format_type == "json":
            return cls._render_json_prompt(analysis)
        elif format_type == "structured_list":
            return cls._render_list_prompt(analysis)
        else:
            return cls._render_markdown_table_prompt(analysis)

    @staticmethod
    def _near_advice(analysis: LinguisticAnalysis) -> str:
        """Caution sentence to append when any token has spelling-near candidates."""
        if any(t.near_entries for t in analysis.tokens):
            return ("Tokens marked '≈' have no exact dictionary entry; the listed candidates are "
                    "spelling-near matches - verify meaning and part of speech against the "
                    "sentence context before using them.")
        return ""

    @classmethod
    def _render_markdown_table_prompt(cls, analysis: LinguisticAnalysis) -> str:
        """Render markdown table format prompt."""
        cs = analysis.clause_structure

        clause_lines = []
        if cs.get("root_predicate"):
            clause_lines.append(f"- **Main Predicate / Action**: {' | '.join(cs['root_predicate'])}")
        if cs.get("subjects"):
            clause_lines.append(f"- **Subject(s)**: {', '.join(cs['subjects'])}")
        if cs.get("objects"):
            clause_lines.append(f"- **Object(s)**: {', '.join(cs['objects'])}")
        if cs.get("obliques_and_adverbials"):
            clause_lines.append(f"- **Adverbial / Obliques**: {', '.join(cs['obliques_and_adverbials'])}")

        clause_str = "\n".join(clause_lines) if clause_lines else "- (Simple clause)"

        # Table rows
        table_rows = [
            "| ID | Word | Lemma | ULY | Morphological Features | Syntactic Role (Dependency) | Dictionary Translation |",
            "|---|---|---|---|---|---|---|"
        ]
        for t in analysis.tokens:
            head_str = f"{t.deprel} -> #{t.head_id} ({t.head_form})"
            table_rows.append(
                f"| {t.id} | {t.form} | {t.lemma} | {t.uly} | {t.morph_summary} | {head_str} | {t.dict_gloss} |"
            )
        table_str = "\n".join(table_rows)

        near_advice = PromptBuilder._near_advice(analysis)
        guidance = "\n".join([
            "1. Preserve all grammatical inflections (tense, mood, voice, polarity, person/number agreements).",
            "2. Resolve pro-drop (omitted subjects) using verbal agreement suffixes where applicable.",
            "3. Use the word-level dictionary definitions in proper contextual syntax.",
            "4. Provide only the final fluent English translation without conversational preamble.",
        ])
        if near_advice:
            guidance += f"\n5. {near_advice}"

        prompt = f"""You are an expert Uyghur-to-English translator and computational linguist.
Translate the following Uyghur sentence into fluent, accurate English using the provided linguistic analysis.

### Uyghur Source Sentence:
{analysis.sentence}

### Clause Architecture & Syntactic Skeleton:
{clause_str}

### Word-by-Word Morphosyntactic Breakdown & Dictionary Entries:
{table_str}

### Translation Guidance:
{guidance}

### English Translation:"""
        return prompt

    @classmethod
    def _render_list_prompt(cls, analysis: LinguisticAnalysis) -> str:
        """Render structured list format prompt."""
        tok_lines = []
        for t in analysis.tokens:
            tok_lines.append(
                f"{t.id}. Word: '{t.form}' | ULY: '{t.uly}' | Lemma: '{t.lemma}' | Features: {t.morph_summary} | "
                f"Syntax: {t.deprel} -> #{t.head_id} {t.head_form} | Dict: {t.dict_gloss}"
            )
        toks_str = "\n".join(tok_lines)

        near_advice = PromptBuilder._near_advice(analysis)
        note = f"\nNote: {near_advice}" if near_advice else ""

        prompt = f"""You are an expert Uyghur-to-English translator and linguist.
Translate the following Uyghur sentence into natural, accurate English using the provided linguistic context.

Sentence (Uyghur):
{analysis.sentence}

Linguistic Annotations:
{toks_str}
{note}
English Translation:"""
        return prompt

    @classmethod
    def _render_json_prompt(cls, analysis: LinguisticAnalysis) -> str:
        """Render JSON structured prompt."""
        data_str = json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)
        near_advice = PromptBuilder._near_advice(analysis)
        note = f"\nNote: {near_advice}" if near_advice else ""
        prompt = f"""You are an expert Uyghur-to-English translator.
Use the structured linguistic context below to produce an accurate English translation.

```json
{data_str}
```
{note}
English Translation:"""
        return prompt
