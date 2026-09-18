#!/usr/bin/env python3
"""
run_pipeline.py - Command Line Interface for Uyghur Linguistic Translation Pipeline
"""

import argparse
import sys
import os
import json
from tqdm import tqdm

from pipeline import (
    PipelineConfig,
    UyghurLinguisticPipeline,
    PipelineResult,
    UyghurDictionary,
)


def run_interactive(pipeline: UyghurLinguisticPipeline, args):
    """Run interactive REPL loop."""
    print("\n" + "=" * 70)
    print("🌍 Uyghur Linguistic Translation Pipeline - Interactive Mode")
    print("Type a Uyghur sentence and press Enter (or 'exit' / 'q' to quit)")
    print("=" * 70 + "\n")

    while True:
        try:
            sent = input("Uyghur> ").strip()
            if not sent:
                continue
            if sent.lower() in ("exit", "quit", "q"):
                print("Exiting...")
                break

            result = pipeline.process_sentence(
                sent,
                translate=args.translate,
                model=args.model
            )

            result.print_summary()

            if args.show_prompt:
                print("\n" + "-" * 30 + " GENERATED PROMPT " + "-" * 30)
                print(result.prompt)
                print("-" * 78 + "\n")

            if args.translate and result.translation:
                print(f"\n🎯 English Translation:\n{result.translation}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)


STAGE_LABELS = {
    "dedup": "dedup — unique reading after Apertium candidate deduplication",
    "rule_based_POS_strong": "MUDT rule POS_strong — copula kept (strong predicate)",
    "rule_based_POS_weak": "MUDT rule POS_weak — copula kept (weak predicate)",
    "rule_based_NEG": "MUDT rule NEG — copula dropped (modifier/argument role)",
    "model_based": "Qwen-2B fine-tuned disambiguation",
    "punct": "punctuation token — no morphological analysis",
    "fallback": "fallback — no Apertium reading aligned to this token",
}


_RULE_LINE = "-" * 66


def print_demo_steps(res: PipelineResult) -> None:
    """Walk one sentence through every pipeline stage, printing each stage's input/output."""
    print("\n" + "=" * 74)
    print(f"📄 DEMO SENTENCE: {res.sentence}")
    print("=" * 74)

    # ── STEP 1: Dependency Parsing ──────────────────────────────────────────────
    print("\n▶ STEP 1/5 — DEPENDENCY PARSING   [MUDT DiaParser · pipeline/parser.py]")
    print(f"   Input : raw Uyghur sentence: \"{res.sentence}\"")
    print("   Output: tokens with syntactic HEAD and DEPREL labels")
    print("   " + _RULE_LINE)
    print(f"   {'ID':>3}  {'FORM':<14s} {'HEAD':>4}  {'DEPREL':<12s} PRED STRENGTH")
    for t in res.parsed_sentence.tokens:
        pred = t.pred_strength if t.pred_strength else "—"
        print(f"   {t.id:>3}  {t.form:<14s} {t.head:>4}  {t.deprel:<12s} {pred}")

    # ── STEP 2: Morphological Tagging & Disambiguation ──────────────────────────
    print("\n▶ STEP 2/5 — MORPHOLOGICAL TAGGING & DISAMBIGUATION")
    print("             [Apertium uig-tagger + MUDT zero-copula rules + Qwen-2B · pipeline/morphology.py]")
    forms = " │ ".join(t.form for t in res.parsed_sentence.tokens)
    print(f"   Input : Step 1 tokens, aligned 1:1 to Apertium readings: {forms}")
    print("   Output: lemma, POS, features per token + stage that resolved the ambiguity")
    print("   " + _RULE_LINE)
    for m in res.morph_tokens:
        src_label = STAGE_LABELS.get(m.disambig_source, m.disambig_source)
        print(f"   [{m.id:2d}] {m.form}  →  {m.natural}")
        print(f"        disambig: {src_label}")
        if m.total_candidates > 1:
            print(f"        ambiguity: {m.total_candidates} candidate reading(s) → kept {len(m.candidates or [m])}")
        if m.candidates and len(m.candidates) > 1:
            print("        candidates (what the disambiguator chose between):")
            for i, c in enumerate(m.candidates, 1):
                chosen = "   ◀ chosen" if c.natural == m.natural else ""
                print(f"          {i}. {c.natural}{chosen}")

    # ── STEP 3: Dictionary Lookup ───────────────────────────────────────────────
    print("\n▶ STEP 3/5 — DICTIONARY LOOKUP   [Schwarz Uyghur→English SQLite · pipeline/dictionary.py]")
    print("   Input : lemmas + POS hints from Step 2 (stem-variant fallbacks: verb '-', terminal devoicing)")
    print("   Output: English gloss per lemma, or [form] when the dictionary misses")
    print("   " + _RULE_LINE)
    for m, entry in zip(res.morph_tokens, res.dict_entries):
        if m.pos == "Punctuation":
            print(f"   [{m.id:2d}] {m.form}  (punctuation — skipped)")
            continue
        is_verb = "Standard verb" in m.features or "verb" in m.pos.lower()
        if entry is None:
            variants = UyghurDictionary.generate_stem_variants(m.lemma, is_verb=is_verb)
            tried = ", ".join(variants)
            print(f"   [{m.id:2d}] {m.form}  lemma={m.lemma} verb={is_verb} → ✗ NONE [tried: {tried}]")
        else:
            print(f"   [{m.id:2d}] {m.form}  lemma={m.lemma} verb={is_verb} → {entry.short_gloss}")

    # ── STEP 4: Analysis & Prompt Assembly ─────────────────────────────────────
    print("\n▶ STEP 4/5 — LINGUISTIC ANALYSIS & PROMPT ASSEMBLY   [pipeline/prompt.py]")
    print("   Input : Step 1 syntax + Step 2 morphology + Step 3 glosses (merged by token ID)")
    print("   Output: clause-structure summary + the in-context LLM prompt")
    cs = res.analysis.clause_structure
    if any(cs.values()):
        print("   Clause structure:")
        for key, vals in cs.items():
            if vals:
                print(f"     {key:<24s} {', '.join(str(v) for v in vals)}")
    print("\n   Merged token breakdown (rows of the prompt's analysis table):")
    print(f"   {'ID':>3}  {'FORM':<14s} {'LEMMA':<12s} {'POS':<10s} SYNTACTIC ROLE        DICTIONARY")
    print("   " + _RULE_LINE)
    for t in res.analysis.tokens:
        syn = f"{t.deprel} -> #{t.head_id} ({t.head_form})"
        print(f"   {t.id:>3}  {t.form:<14s} {t.lemma:<12s} {t.pos:<10s} {syn:<22s} {t.dict_gloss}")
    print("\n   Generated in-context prompt (Step 5 input):")
    print("   " + _RULE_LINE)
    print(res.prompt)

    # ── STEP 5: Downstream LLM Translation ─────────────────────────────────────
    print("\n▶ STEP 5/5 — DOWNSTREAM LLM TRANSLATION   [pipeline/translator.py]")
    print("   Input : the prompt rendered in Step 4")
    if res.translation is not None:
        print(f"   Output: 🎯 {res.translation}")
    else:
        print("   Output: (skipped — rerun with --translate to call the downstream LLM)")



def run_demo(pipeline: UyghurLinguisticPipeline, args):
    """Run demonstration across representative linguistic cases, printing every pipeline step."""
    demo_sentences = [
        # # 1. Simple nominal predicate with copula
        # "بۇ بىر ياخشى كىتاب .",
        # # 2. Transitive action with dative goal
        # "ئۇ مەكتەپكە باردى .",
        # # 3. First person plural possessive and past tense verb
        # "بىز تۈنۈگۈن كونا دوستىمىزنى كۆردۇق .",
        # 4. Complex sentence with subordinate clause from FLORES
        "دۈشەنبە كۈنى، ستانفورد ئۇنىۋېرسىتېتى تېببىي ئىنستىتۇتىنىڭ ئالىملىرى ھۈجەيرە تۈرلەيدىغان دىياگنوز قورالىنىڭ كەشىپ قىلىنغانلىقىنى ئېلان قىلدى، ئۇ ئادەتتىكى سىياھ پۈركۈيدىغان پىرىنتېر ئارقىلىقلا بېسىپ چىقارغىلى بولىدىغان ئۆزەك بولۇپ، ھەر بىرى ئۈچۈن تەخمىنەن بىر ئامېرىكا سېنتى چىقىم كېتىدۇ."
    ]

    print("\n" + "=" * 74)
    print("🚀 RUNNING DEMO ON REPRESENTATIVE UYGHUR SENTENCES")
    print("   Each sample walks through all pipeline stages with their inputs and outputs:")
    print("   1. Dependency Parsing → 2. Morphology & Disambiguation → 3. Dictionary")
    print("   → 4. Analysis & Prompt → 5. LLM Translation (--translate)")
    print("=" * 74)

    results = pipeline.process_batch(
        demo_sentences,
        translate=args.translate,
        model=args.model
    )

    for i, res in enumerate(results, 1):
        print(f"\n[DEMO SAMPLE {i}/{len(demo_sentences)}]")
        print_demo_steps(res)


def run_batch_file(pipeline: UyghurLinguisticPipeline, args):
    """Process a batch text file (one sentence per line)."""
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"📖 Reading sentences from: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print("⚠️ Input file is empty.")
        return

    print(f"Processing {len(lines)} sentences in batches of {args.batch_size}...")
    output_rows = []

    for i in tqdm(range(0, len(lines), args.batch_size), desc="Processing"):
        batch = lines[i:i + args.batch_size]
        results = pipeline.process_batch(
            batch,
            translate=args.translate,
            model=args.model
        )
        for r in results:
            output_rows.append(r.to_dict())

    out_file = args.output_file or (os.path.splitext(args.input_file)[0] + ".output.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Finished! Output written to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Uyghur Linguistic Translation Pipeline CLI"
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str,
        default=None,
        help="Input a single Uyghur sentence to process"
    )
    parser.add_argument(
        "--input_file", "-f",
        type=str,
        default=None,
        help="Path to input text file (one sentence per line) or JSONL"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo on representative Uyghur sample sentences"
    )
    parser.add_argument(
        "--translate", "-t",
        action="store_true",
        help="Execute downstream LLM translation"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Downstream LLM model identifier (e.g. 'openai/gpt-4o', 'google/gemini-2.5-flash')"
    )
    parser.add_argument(
        "--prompt_format",
        choices=["markdown_table", "structured_list", "json"],
        default="markdown_table",
        help="Format style for generated prompt"
    )
    parser.add_argument(
        "--show_prompt", "-p",
        action="store_true",
        help="Print the generated in-context prompt to stdout"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda:0' or 'cpu')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for batch file processing"
    )

    args = parser.parse_args()

    # Build Configuration
    config = PipelineConfig()
    if args.device:
        config.device = args.device
    if args.prompt_format:
        config.prompt_format = args.prompt_format
    if args.model:
        config.llm_model = args.model

    print("🔧 Initializing Uyghur Linguistic Pipeline...")
    pipeline = UyghurLinguisticPipeline(config=config, load_models=True)
    print("✅ Pipeline ready!\n")

    if args.sentence:
        result = pipeline.process_sentence(
            args.sentence,
            translate=args.translate,
            model=args.model
        )
        result.print_summary()
        if args.show_prompt:
            print("\n" + "-" * 30 + " GENERATED PROMPT " + "-" * 30)
            print(result.prompt)
            print("-" * 78 + "\n")
        if args.translate and result.translation:
            print(f"\n🎯 English Translation:\n{result.translation}\n")

    elif args.input_file:
        run_batch_file(pipeline, args)

    elif args.demo:
        run_demo(pipeline, args)

    elif args.interactive:
        run_interactive(pipeline, args)

    else:
        # If no arguments given, run demo
        print("No input arguments provided. Running demo mode by default...")
        args.show_prompt = True
        run_demo(pipeline, args)


if __name__ == "__main__":
    main()
