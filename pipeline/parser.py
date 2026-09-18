"""
parser.py - Dependency Parser module using MUDT-trained DiaParser
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union
import torch

# Patch PyTorch 2.6+ weights_only issue with DiaParser checkpoints
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from diaparser.parsers import Parser

# Uyghur, Arabic, and standard punctuation set
PUNC_REGEX = r"([،۔。！？!?\.\,\:\；;\"“”‘’\(\)\[\]\{\}…—\-–«»‹›؟])"


@dataclass
class ParsedToken:
    """Represents a single parsed token from dependency parsing."""
    id: int
    form: str
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: str = "_"
    head: int = 0
    deprel: str = "dep"
    pred_strength: Optional[str] = None  # 'strong' (head of cop:zero), 'weak' (root), or None

    def to_conllu_line(self) -> str:
        """Serialize token to standard CoNLL-U line."""
        return f"{self.id}\t{self.form}\t{self.lemma}\t{self.upos}\t{self.xpos}\t{self.feats}\t{self.head}\t{self.deprel}\t_\t_"


@dataclass
class ParsedSentence:
    """Represents a full dependency-parsed sentence."""
    text: str
    tokens: List[ParsedToken] = field(default_factory=list)

    @property
    def token_forms(self) -> List[str]:
        return [t.form for t in self.tokens]

    def get_token(self, token_id: int) -> Optional[ParsedToken]:
        for t in self.tokens:
            if t.id == token_id:
                return t
        return None

    def get_head_token(self, token: ParsedToken) -> Optional[ParsedToken]:
        if token.head == 0:
            return None
        return self.get_token(token.head)

    def to_conllu(self, sent_id: str = "1") -> str:
        """Export sentence as CoNLL-U format."""
        lines = [
            f"# sent_id = {sent_id}",
            f"# text = {self.text}"
        ]
        for t in self.tokens:
            lines.append(t.to_conllu_line())
        return "\n".join(lines)


class DependencyParser:
    """
    Dependency Parser wrapper for MUDT-trained XLM-RoBERTa DiaParser model.
    """

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.device = device
        self._load_parser()

    def _load_parser(self):
        # Convert device string to int if cuda (e.g. 'cuda:0' -> 0)
        dev_id = -1
        if "cuda" in self.device:
            if ":" in self.device:
                try:
                    dev_id = int(self.device.split(":")[-1])
                except ValueError:
                    dev_id = 0
            else:
                dev_id = 0
        self.parser = Parser.load(self.model_dir, device=dev_id)

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        """Tokenize sentence by isolating punctuation and whitespace."""
        s = re.sub(PUNC_REGEX, r" \1 ", sentence)
        s = re.sub(r"\s+", " ", s).strip()
        return s.split(" ") if s else []

    @staticmethod
    def mark_predicates(tokens: List[ParsedToken]) -> List[ParsedToken]:
        """
        Mark syntactic predicate strengths:
        - STRONG: token is HEAD of a `cop:zero` arc.
        - WEAK: token is root of clause (if not strong).
        """
        cop_zero_heads = {t.head for t in tokens if t.deprel.split(':')[0] == 'cop' and 'zero' in t.deprel}
        for t in tokens:
            if t.id in cop_zero_heads:
                t.pred_strength = "strong"
            elif t.deprel == "root" or t.head == 0:
                t.pred_strength = "weak"
            else:
                t.pred_strength = None
        return tokens

    def parse_sentence(self, sentence: str) -> ParsedSentence:
        """Parse a single Uyghur sentence."""
        tokens = self.tokenize(sentence)
        if not tokens:
            return ParsedSentence(text=sentence, tokens=[])

        dataset = self.parser.predict([tokens], tree=True)
        sent = dataset.sentences[0]

        ids = [int(x) for x in sent.values[0]]
        forms = list(sent.values[1])
        heads = [int(x) for x in sent.values[6]]
        deprels = list(sent.values[7])

        parsed_tokens = []
        for tid, form, head, deprel in zip(ids, forms, heads, deprels):
            parsed_tokens.append(
                ParsedToken(
                    id=tid,
                    form=form,
                    head=head,
                    deprel=deprel
                )
            )

        self.mark_predicates(parsed_tokens)
        return ParsedSentence(text=sentence, tokens=parsed_tokens)

    def parse_batch(self, sentences: List[str]) -> List[ParsedSentence]:
        """Parse a batch of Uyghur sentences efficiently."""
        token_lists = [self.tokenize(s) for s in sentences]
        valid_indices = [i for i, toks in enumerate(token_lists) if len(toks) > 0]
        valid_toks = [token_lists[i] for i in valid_indices]

        results = [ParsedSentence(text=s, tokens=[]) for s in sentences]
        if not valid_toks:
            return results

        dataset = self.parser.predict(valid_toks, tree=True)

        for orig_idx, sent in zip(valid_indices, dataset.sentences):
            ids = [int(x) for x in sent.values[0]]
            forms = list(sent.values[1])
            heads = [int(x) for x in sent.values[6]]
            deprels = list(sent.values[7])

            parsed_tokens = []
            for tid, form, head, deprel in zip(ids, forms, heads, deprels):
                parsed_tokens.append(
                    ParsedToken(
                        id=tid,
                        form=form,
                        head=head,
                        deprel=deprel
                    )
                )

            self.mark_predicates(parsed_tokens)
            results[orig_idx] = ParsedSentence(text=sentences[orig_idx], tokens=parsed_tokens)

        return results
