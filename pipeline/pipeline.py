"""
pipeline.py - Master Pipeline Orchestrator for Uyghur Linguistic Extraction and Translation
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

from .config import PipelineConfig
from .parser import DependencyParser, ParsedSentence, ParsedToken
from .morphology import MorphologicalTagger, MorphToken
from .dictionary import UyghurDictionary, DictEntry
from .prompt import PromptBuilder, LinguisticAnalysis
from .translator import LLMTranslator


@dataclass
class PipelineResult:
    """Represents the complete pipeline execution result for a single sentence."""
    sentence: str
    parsed_sentence: ParsedSentence
    morph_tokens: List[MorphToken]
    dict_entries: List[Optional[DictEntry]]
    analysis: LinguisticAnalysis
    prompt: str
    translation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "sentence": self.sentence,
            "translation": self.translation,
            "linguistic_analysis": self.analysis.to_dict(),
            "prompt": self.prompt,
            "conllu": self.parsed_sentence.to_conllu()
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def print_summary(self):
        """Print a visually clean summary to terminal."""
        print("\n" + "=" * 70)
        print("🔍 UYGHUR LINGUISTIC PIPELINE ANALYSIS")
        print("=" * 70)
        print(f"Source: {self.sentence}")
        if self.translation:
            print(f"Translation: {self.translation}")
        print("-" * 70)
        print("TOKEN BREAKDOWN:")
        for t in self.analysis.tokens:
            print(f"  [{t.id:2d}] {t.form:15s} | Lemma: {t.lemma:12s} | POS: {t.pos:12s} | "
                  f"Syntax: {t.deprel:8s} -> #{t.head_id:<2d} | Dict: {t.dict_gloss}")
        print("=" * 70)


class UyghurLinguisticPipeline:
    """
    End-to-end Pipeline:
    Input Sentence (Uyghur)
            ↓
    [1] Dependency Parser     →  HEAD, DEPREL
    [2] Morphological Tagger  →  LEMMA, XPOS, FEATS (Apertium + MUDT Rules + Qwen 2B)
    [3] Dictionary Lookup     →  word-level translations (Schwarz SQLite DB)
            ↓
    Linguistic description (in-context prompt)
            ↓
    LLM → English translation
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        load_models: bool = True
    ):
        self.config = config or PipelineConfig()
        self.load_models = load_models

        # Initialize sub-components
        self.parser: Optional[DependencyParser] = None
        self.morphology: Optional[MorphologicalTagger] = None
        self.dictionary: Optional[UyghurDictionary] = None
        self.translator: Optional[LLMTranslator] = None
        self.prompt_builder = PromptBuilder()

        if load_models:
            self._initialize_components()

    def _initialize_components(self):
        """Load and initialize all components."""
        # 1. Dependency Parser
        if self.parser is None:
            self.parser = DependencyParser(
                model_dir=self.config.mudt_model_dir,
                device=self.config.device
            )

        # 2. Morphological Tagger & Disambiguator
        if self.morphology is None:
            self.morphology = MorphologicalTagger(
                apertium_dir=self.config.apertium_dir,
                disambiguater_model_dir=self.config.disambiguater_model_dir,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                load_model=True
            )

        # 3. Dictionary
        if self.dictionary is None:
            self.dictionary = UyghurDictionary(
                db_path=self.config.dictionary_db_path
            )

        # 4. Translator
        if self.translator is None:
            self.translator = LLMTranslator(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_api_base,
                default_model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )

    def process_sentence(
        self,
        sentence: str,
        translate: bool = False,
        model: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a single sentence through the full pipeline.
        """
        return self.process_batch([sentence], translate=translate, model=model)[0]

    def process_batch(
        self,
        sentences: List[str],
        translate: bool = False,
        model: Optional[str] = None
    ) -> List[PipelineResult]:
        """
        Process a batch of sentences through the full pipeline.
        """
        self._initialize_components()

        # Step 1: Dependency Parsing (HEAD, DEPREL)
        parsed_sentences = self.parser.parse_batch(sentences)

        # Step 2: Morphological Tagging & Disambiguation (LEMMA, XPOS, FEATS)
        all_morph_tokens = self.morphology.tag_batch(parsed_sentences)

        # Step 3: Dictionary Lookup (Word-level translations)
        results: List[PipelineResult] = []

        prompts_to_translate = []
        indices_to_translate = []

        for idx, (ps, morphs) in enumerate(zip(parsed_sentences, all_morph_tokens)):
            # Lookup stems in dictionary
            dict_entries: List[Optional[DictEntry]] = []
            near_entries: List[List[DictEntry]] = []
            for m in morphs:
                if m.pos == "Punctuation":
                    dict_entries.append(None)
                    near_entries.append([])
                else:
                    is_verb = ("Standard verb" in m.features or "verb" in m.pos.lower())
                    entry = self.dictionary.lookup(
                        m.lemma, is_verb=is_verb, pos_hint=m.pos, surface=m.form
                    )
                    dict_entries.append(entry)
                    if entry is None and m.pos != "Proper noun":
                        # spelling-near context candidates (with verify caveat);
                        # skipped for proper nouns, whose near matches are noise.
                        near_entries.append(self.dictionary.lookup_near(m.lemma, m.form))
                    else:
                        near_entries.append([])

            # Step 4: Build Linguistic Analysis and In-Context Prompt
            analysis = self.prompt_builder.build_analysis(
                ps, morphs, dict_entries, near_entries=near_entries
            )
            prompt = self.prompt_builder.build_prompt(
                analysis,
                format_type=self.config.prompt_format
            )

            res = PipelineResult(
                sentence=ps.text,
                parsed_sentence=ps,
                morph_tokens=morphs,
                dict_entries=dict_entries,
                analysis=analysis,
                prompt=prompt,
                translation=None
            )
            results.append(res)

            if translate:
                prompts_to_translate.append(prompt)
                indices_to_translate.append(idx)

        # Step 5: Downstream LLM Translation (if requested)
        if translate and prompts_to_translate:
            for p_idx, prompt in zip(indices_to_translate, prompts_to_translate):
                translation = self.translator.translate(prompt, model=model)
                results[p_idx].translation = translation

        return results

    def translate_sentence(
        self,
        sentence: str,
        model: Optional[str] = None
    ) -> str:
        """Convenience method to translate a single Uyghur sentence directly."""
        res = self.process_sentence(sentence, translate=True, model=model)
        return res.translation or ""

    def translate_batch(
        self,
        sentences: List[str],
        model: Optional[str] = None
    ) -> List[str]:
        """Convenience method to translate a batch of Uyghur sentences."""
        results = self.process_batch(sentences, translate=True, model=model)
        return [r.translation or "" for r in results]
