"""
morphology.py - Morphological Tagger and Disambiguator using Apertium and fine-tuned Qwen model
"""

import os
import io
import re
import difflib
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from streamparser import parse_file, reading_to_string

from .parser import ParsedSentence, ParsedToken

# =============================================================================
# TAG -> GLOSS MAPPINGS (Apertium symbols -> UD-style human-readable glosses)
# =============================================================================
MAP = {
    'post': 'Postposition', 'lpar': 'Left parenthesis', 'dem': 'Demonstrative', 'f': 'Feminine', 'al': 'Altres',
    'gpr_fut': 'Future verbal adjective', 'nom': 'Nominative', 'ger_past': 'Past gerund', 'ger_fut': 'Future gerund',
    'cop': 'Copula', 'np': 'Proper noun', 'acc': 'Accusative', 'v': 'Standard verb', 'loc': 'Locative',
    'cnjadv': 'Conjunctive adverb', 'adj': 'Adjective', 'gpr_impf': 'Imperfect verbal adjective',
    'imp': 'Imperative', 'gna_impf': 'Imperfect verbal adverb', 'def': 'Definite', 'num': 'Numeral',
    'cnjcoo': 'Co-ordinating conjunction', 'n': 'Noun', 'aor': 'Aorist', 'neg': 'Negative', 'vaux': 'Auxiliary verb',
    'postadv': 'Post-adverb', 'abbr': "Abbreviation", 'lquot': 'Left quote', 'abl': 'Ablative',
    'prn': 'Pronoun', 'mod': 'Modal word', 'gna_perf': 'Perfect verbal adverb', 'cog': 'Cognomen', 'dat': 'Dative',
    'qst': 'Interrogative particle', 'frm': 'Formal', 'ger_impf': 'Imperfect gerund',
    'gpr_past': 'Past verbal adjective', 'attr': 'Attributive', 'gen': 'Genitive', 'fut': 'Future',
    'ant': 'Anthroponym', 'cnjsub': 'Sub-ordinating conjunction', 'sent': 'Sentence-ending punctuation',
    'ger': 'Gerund', 'cm': 'Comma punctuation', 'adv': 'Adverb', 'rquot': 'Right quote', 'ifi': 'Past definite',
    'ins': 'Instrumental or Instructive', 'iv': 'Intransitive', 'rpar': 'Right parenthesis', 'ij': 'Interjection',
    'comp': 'Comparative', 'coll': 'Collective', 'itg': 'Interrogative', 'det': 'Determiner', 'guio': 'Hyphen',
    'ind': 'Indefinite', 'ger_perf': 'Perfect gerund', 'px2pl': 'Second person plural possessive',
    'pat': 'Patronymic', 'pass': 'Passive voice', 'tv': 'Transitive', 'prc_perf': 'Perfect participle',
    'pres': 'Present', 'px1sg': 'First person singular possessive', 'p2': 'Second person',
    'px3pl': 'Third person plural possessive', 'px3sg': 'Third person singular possessive', 'p3': 'Third person',
    'top': 'Toponym', 'sg': 'Singular', 'qnt': 'Quantifier', 'mf': 'Masculine or feminine', 'm': 'Masculine',
    'percent': 'Percentage', 'prc_impf': 'Imperfect participle', 'px1pl': 'First person plural possessive',
    'p1': 'First person', 'past': 'Past', 'pers': 'Personal', 'pl': 'Plural', 'org': 'Organisation',
    'px3sp': 'Third person possessive singular or plural', 'ref': 'Reflexive', 'ord': 'Ordinal',
    'px2sg': 'Second person singular possessive', 'advl': 'Adverbial', 'subst': 'Substantive'
}

TURKISH_SPECIFIC_MAP = {
    'gpr_rsub': 'Relative substantival verbal adjective', 'ger_pabs': 'Past absolute gerund',
    'ger_inf': 'Infinitive gerund', 'dub': 'Dubitative', 'abil': 'Abilitative', 'gna_neg': 'Negative verbal adverb',
    'dek': 'Equative', 'evid': 'Evidential', 'aorp': 'Aorist participle', 'gna_cond': 'Conditional verbal adverb',
    'iver': 'Continuative aspect', 'ter': 'Terminative converb', 'che': 'Limitative', 'td': 'Unspecified transitivity'
}
MAP.update(TURKISH_SPECIFIC_MAP)
GARBAGE_TAGS = {'err_orth'}

DELIMITER = "SENTBOUNDARY"

MODIFIER_DEPRELS = {
    'amod', 'nummod', 'det', 'case', 'nmod', 'obj', 'obl', 'advmod', 'mark', 'nsubj', 'csubj', 'iobj'
}


def gloss_tag(tag: str) -> Optional[str]:
    """Map raw Apertium tag to standardized readable gloss."""
    t = tag.lower()
    if t in GARBAGE_TAGS or tag in GARBAGE_TAGS:
        return None
    if t in MAP:
        return MAP[t]
    base = re.sub(r'\d+$', '', t)
    if base != t and base in MAP:
        return MAP[base]
    return t.upper()


@dataclass
class MorphReading:
    """Represents a single parsed candidate analysis."""
    natural: str
    feats: List[str]
    has_cop: bool
    lemma: str
    pos: str
    raw: str
    enclitics: List[str] = field(default_factory=list)


@dataclass
class MorphToken:
    """Represents the morphological analysis for a token in context."""
    id: int
    form: str
    lemma: str
    pos: str
    features: List[str] = field(default_factory=list)
    natural: str = ""
    enclitics: List[str] = field(default_factory=list)
    disambig_source: str = "fallback"  # 'dedup', 'rule_based', 'model_based', 'punct', 'fallback'
    candidates: List[MorphReading] = field(default_factory=list)
    total_candidates: int = 1


def parse_reading(reading: Any) -> Optional[MorphReading]:
    """Parse an Apertium reading object into a structured MorphReading."""
    raw = reading_to_string(reading)
    parts = raw.split('+')
    natural_parts, feats = [], []
    enclitics = []
    main_lemma = ""
    main_pos = "Unknown"
    main_ok = False

    for pidx, part in enumerate(parts):
        if '<' not in part:
            continue
        lemma = part[:part.index('<')]
        tags = part[part.index('<') + 1: part.rindex('>')].split('><')
        gloss = [g for g in (gloss_tag(t) for t in tags) if g is not None]

        if pidx == 0:
            if not lemma.strip() or not gloss:
                return None
            main_ok = True
            main_lemma = lemma
            main_pos = gloss[0]
            natural_parts.append(f"Lemma: {lemma} | Features: {', '.join(gloss)}")
            feats.extend(gloss)
        else:
            if not gloss:
                continue
            atomic = f"Lemma: {lemma} | Features: {', '.join(gloss)}"
            natural_parts.append(atomic)
            enclitics.append(atomic)
            feats.append(atomic)

    if not main_ok:
        return None

    return MorphReading(
        natural=" + Combined with: ".join(natural_parts),
        feats=feats,
        has_cop=("<cop>" in raw),
        lemma=main_lemma,
        pos=main_pos,
        raw=raw,
        enclitics=enclitics
    )


def dedup_candidates(readings: List[Any]) -> List[MorphReading]:
    """Deduplicate candidate readings by normalized feature set."""
    unique, seen = [], set()
    for r in readings:
        parsed = parse_reading(r)
        if parsed is None:
            continue
        key = frozenset(parsed.feats)
        if key in seen:
            continue
        seen.add(key)
        unique.append(parsed)
    return unique


# =============================================================================
# ALIGNMENT HELPERS
# =============================================================================
def filter_and_flatten(tokens: List[str], split_char: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    orig_filtered, flat_list = [], []
    for true_idx, tok in enumerate(tokens):
        if not any(c.isalnum() for c in tok):
            continue
        orig_filtered.append(tok)
        for sw in tok.split(split_char):
            if sw.strip():
                flat_list.append({"orig_idx": true_idx, "subword": sw})
    return orig_filtered, flat_list


def get_components(pairs: List[Tuple[int, int]]) -> List[Tuple[List[int], List[int]]]:
    components, cg, ca = [], set(), set()
    for g, a in pairs:
        if cg and g not in cg and a not in ca:
            components.append((sorted(cg), sorted(ca)))
            cg, ca = {g}, {a}
        else:
            cg.add(g)
            ca.add(a)
    if cg or ca:
        components.append((sorted(cg), sorted(ca)))
    return components


class MorphologicalTagger:
    """
    Hybrid Morphological Tagger and Disambiguator.
    Combines Apertium rule-based generator, MUDT zero-copula syntactic filters,
    and fine-tuned Qwen-2B causal LM disambiguator.
    """

    def __init__(
        self,
        apertium_dir: str,
        disambiguater_model_dir: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_model: bool = True
    ):
        self.apertium_dir = apertium_dir
        self.disambiguater_model_dir = disambiguater_model_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

        if load_model:
            self._load_disambiguater_model()

    def _load_disambiguater_model(self):
        """Load fine-tuned Qwen 2B model for disambiguation."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.disambiguater_model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.disambiguater_model_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.model.eval()

    def run_apertium_batched(self, texts: List[str]) -> List[List[Any]]:
        """Run Apertium uig-tagger on a batch of sentences."""
        full_input = f" {DELIMITER} ".join(texts)
        cmd = ["apertium", "-d", self.apertium_dir, "uig-tagger"]
        res = subprocess.run(
            cmd,
            input=full_input.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            # Fallback 1-by-1
            return [self._run_apertium_single(t) for t in texts]

        stream = io.StringIO(res.stdout.decode("utf-8"))
        sentences_lus, current = [], []
        for _, lu in parse_file(stream, with_text=True):
            if lu is None:
                continue
            if lu.wordform == DELIMITER:
                sentences_lus.append(current)
                current = []
            else:
                current.append(lu)
        if current:
            sentences_lus.append(current)

        if len(sentences_lus) != len(texts):
            # Batch size mismatch fallback
            return [self._run_apertium_single(t) for t in texts]

        return sentences_lus

    def _run_apertium_single(self, text: str) -> List[Any]:
        cmd = ["apertium", "-d", self.apertium_dir, "uig-tagger"]
        res = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            return []
        stream = io.StringIO(res.stdout.decode("utf-8"))
        return [lu for _, lu in parse_file(stream, with_text=True) if lu is not None]

    @staticmethod
    def apply_mudt_rules(
        candidates: List[MorphReading],
        parsed_tokens: List[ParsedToken]
    ) -> Tuple[List[MorphReading], Optional[str]]:
        """
        Apply MUDT zero-copula syntactic disambiguation rules.
        """
        n = len(candidates)
        if n <= 1:
            return candidates, None

        deprels = {t.deprel.split(':')[0] for t in parsed_tokens}
        has_cop_cand = any(c.has_cop for c in candidates)
        is_strong = any(t.pred_strength == 'strong' for t in parsed_tokens)
        is_root = 'root' in deprels or any(t.pred_strength == 'weak' for t in parsed_tokens)

        # Rule 1b POSITIVE (Predicate -> keep only copula)
        if (is_strong or is_root) and has_cop_cand:
            filtered = [c for c in candidates if c.has_cop]
            if filtered and len(filtered) < n:
                strength = 'strong' if is_strong else 'weak'
                return filtered, f"POS_{strength}"

        # Rule 1a NEGATIVE (Modifier/Argument -> drop copula)
        elif (deprels & MODIFIER_DEPRELS) and has_cop_cand:
            filtered = [c for c in candidates if not c.has_cop]
            if filtered and len(filtered) < n:
                return filtered, "NEG"

        return candidates, None

    @staticmethod
    def _mark_target_in_sentence(sentence: str, target_word: str, occurrence: int = 0) -> str:
        tw = target_word.strip()
        if not tw:
            return sentence
        pattern = re.compile(rf"(?<!\S){re.escape(tw)}(?!\S)")
        matches = list(pattern.finditer(sentence))
        if matches:
            m = matches[occurrence] if occurrence < len(matches) else matches[0]
            return sentence[:m.start()] + f"<t> {tw} </t>" + sentence[m.end():]
        if tw in sentence:
            return sentence.replace(tw, f"<t> {tw} </t>", 1)
        return f"{sentence} <t> {tw} </t>"

    def disambiguate_with_qwen_batch(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Run batched inference with Qwen-2B to disambiguate remaining ambiguous tokens.
        Each task dict contains:
          - 'sentence': str
          - 'target_word': str
          - 'occurrence': int
          - 'candidates': List[MorphReading]
        Returns list of chosen candidate indices (0-indexed).
        """
        if not tasks or self.model is None or self.tokenizer is None:
            return [0] * len(tasks)

        prompts = []
        for t in tasks:
            sent_marked = self._mark_target_in_sentence(
                t['sentence'], t['target_word'], t.get('occurrence', 0)
            )
            cands_str = "\n".join(
                f"{i + 1}. {c.natural}" for i, c in enumerate(t['candidates'])
            )
            prompt_content = (
                "You are an expert computational linguist specializing in Uyghur morphosyntax and contextual disambiguation.\n"
                "Disambiguate the marked Uyghur word by choosing the analysis whose features are ALL correct in context.\n"
                f"Sentence: {sent_marked}\n"
                f"Word: {t['target_word']}\n"
                f"Candidates:\n{cands_str}"
            )
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(rendered)

        # Batch encode with left-padding for causal LM generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        results = []
        bs = 16
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i:i + bs]
            batch_tasks = tasks[i:i + bs]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False
                )

            for j, task in enumerate(batch_tasks):
                input_len = inputs["input_ids"][j].shape[0]
                gen_text = self.tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
                
                # Parse ANSWER: \d+
                m = re.findall(r"ANSWER:\s*(\d+)", gen_text)
                if m:
                    idx = int(m[-1]) - 1
                else:
                    m2 = re.findall(r"\d+", gen_text)
                    idx = (int(m2[-1]) - 1) if m2 else 0

                num_cands = len(task['candidates'])
                if 0 <= idx < num_cands:
                    results.append(idx)
                else:
                    results.append(0)

        return results

    def tag_batch(
        self,
        parsed_sentences: List[ParsedSentence]
    ) -> List[List[MorphToken]]:
        """
        Perform end-to-end morphological tagging and disambiguation on a batch of parsed sentences.
        """
        texts = [ps.text for ps in parsed_sentences]
        apt_batch = self.run_apertium_batched(texts)

        all_sentence_morphs: List[List[MorphToken]] = []
        qwen_tasks = []
        task_pointers = []  # (sentence_idx, token_idx)

        for s_idx, (ps, apt_lus) in enumerate(zip(parsed_sentences, apt_batch)):
            g_tokens = ps.tokens
            morph_tokens: List[MorphToken] = []

            # Wordform occurrence tracking
            wf_counts: Dict[str, int] = {}
            occ_map: Dict[int, int] = {}
            for idx, t in enumerate(g_tokens):
                wf = t.form
                occ_map[idx] = wf_counts.get(wf, 0)
                wf_counts[wf] = wf_counts.get(wf, 0) + 1

            # Build alignments
            gf, gflat = filter_and_flatten([t.form for t in g_tokens], "_")
            af, aflat = filter_and_flatten([lu.wordform for lu in apt_lus], " ")

            matched_apt_for_g: Dict[int, List[Any]] = {i: [] for i in range(len(g_tokens))}

            if gflat and aflat:
                sm = difflib.SequenceMatcher(
                    None, [x["subword"] for x in gflat], [x["subword"] for x in aflat]
                )
                pairs = []
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        pairs.extend(zip(range(i1, i2), range(j1, j2)))

                comps = get_components([(gflat[x]["orig_idx"], aflat[y]["orig_idx"]) for x, y in pairs])
                for g_indices, a_indices in comps:
                    for gi in g_indices:
                        matched_apt_for_g[gi] = [apt_lus[ai] for ai in a_indices]

            # Process each token
            for t_idx, token in enumerate(g_tokens):
                # Pure punctuation check
                if not any(c.isalnum() for c in token.form):
                    morph_tokens.append(
                        MorphToken(
                            id=token.id,
                            form=token.form,
                            lemma=token.form,
                            pos="Punctuation",
                            features=["Punctuation"],
                            natural=f"Lemma: {token.form} | Features: Punctuation",
                            disambig_source="punct",
                            candidates=[],
                            total_candidates=1
                        )
                    )
                    continue

                lus = matched_apt_for_g.get(t_idx, [])
                if not lus:
                    # Fallback for unaligned or unknown token
                    morph_tokens.append(
                        MorphToken(
                            id=token.id,
                            form=token.form,
                            lemma=token.form,
                            pos="Noun",
                            features=["Noun"],
                            natural=f"Lemma: {token.form} | Features: Noun",
                            disambig_source="fallback",
                            candidates=[],
                            total_candidates=1
                        )
                    )
                    continue

                # Collect all readings from matched LUs
                raw_readings = []
                for lu in lus:
                    raw_readings.extend(lu.readings)

                cands = dedup_candidates(raw_readings)
                if not cands:
                    # Fallback
                    morph_tokens.append(
                        MorphToken(
                            id=token.id,
                            form=token.form,
                            lemma=token.form,
                            pos="Noun",
                            features=["Noun"],
                            natural=f"Lemma: {token.form} | Features: Noun",
                            disambig_source="fallback",
                            candidates=[],
                            total_candidates=1
                        )
                    )
                    continue

                if len(cands) == 1:
                    chosen = cands[0]
                    morph_tokens.append(
                        MorphToken(
                            id=token.id,
                            form=token.form,
                            lemma=chosen.lemma,
                            pos=chosen.pos,
                            features=chosen.feats,
                            natural=chosen.natural,
                            enclitics=chosen.enclitics,
                            disambig_source="dedup",
                            candidates=cands,
                            total_candidates=1
                        )
                    )
                    continue

                # Apply MUDT Rules
                filtered_cands, rule_tag = self.apply_mudt_rules(cands, [token])
                if len(filtered_cands) == 1:
                    chosen = filtered_cands[0]
                    morph_tokens.append(
                        MorphToken(
                            id=token.id,
                            form=token.form,
                            lemma=chosen.lemma,
                            pos=chosen.pos,
                            features=chosen.feats,
                            natural=chosen.natural,
                            enclitics=chosen.enclitics,
                            disambig_source=f"rule_based_{rule_tag}",
                            candidates=filtered_cands,
                            total_candidates=len(cands)
                        )
                    )
                    continue

                # Still ambiguous -> Queue for Qwen model
                surviving = filtered_cands if filtered_cands else cands
                qwen_tasks.append({
                    "sentence": ps.text,
                    "target_word": token.form,
                    "occurrence": occ_map.get(t_idx, 0),
                    "candidates": surviving
                })
                # Temporary placeholder
                morph_tokens.append(
                    MorphToken(
                        id=token.id,
                        form=token.form,
                        lemma=surviving[0].lemma,
                        pos=surviving[0].pos,
                        features=surviving[0].feats,
                        natural=surviving[0].natural,
                        enclitics=surviving[0].enclitics,
                        disambig_source="model_based",
                        candidates=surviving,
                        total_candidates=len(cands)
                    )
                )
                task_pointers.append((s_idx, len(morph_tokens) - 1, surviving))

            all_sentence_morphs.append(morph_tokens)

        # Run batched Qwen disambiguation if any ambiguous words remain
        if qwen_tasks and self.model is not None:
            chosen_indices = self.disambiguate_with_qwen_batch(qwen_tasks)
            for (s_idx, tok_idx, cands), chosen_idx in zip(task_pointers, chosen_indices):
                chosen = cands[chosen_idx]
                tok = all_sentence_morphs[s_idx][tok_idx]
                tok.lemma = chosen.lemma
                tok.pos = chosen.pos
                tok.features = chosen.feats
                tok.natural = chosen.natural
                tok.enclitics = chosen.enclitics
                tok.disambig_source = "model_based"

        return all_sentence_morphs

    def tag_sentence(self, parsed_sentence: ParsedSentence) -> List[MorphToken]:
        """Perform morphological tagging on a single parsed sentence."""
        return self.tag_batch([parsed_sentence])[0]
