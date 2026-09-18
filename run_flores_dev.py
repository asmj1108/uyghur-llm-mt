#!/usr/bin/env python3
"""
run_flores_dev.py - Run the FLORES-200 dev set through the Uyghur linguistic
pipeline and collect inspection statistics.

Outputs (in --output_dir, default `stats/flores_dev/`):
  <split>.results.jsonl  - one record per sentence (tokens, glosses, translation)
  <split>.stats.json     - aggregate statistics for pipeline improvement
  and prints a summary to stdout.

Examples:
  # Full pipeline incl. LLM translation against a local OpenAI-compatible server
  python run_flores_dev.py --model Qwen/Qwen2.5-7B-Instruct

  # Analysis only, no LLM calls
  python run_flores_dev.py --no-translate

  # Quick smoke test
  python run_flores_dev.py --limit 8 --no-translate
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

from tqdm import tqdm

from pipeline import PipelineConfig, UyghurLinguisticPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLORES_DIR = os.path.join(BASE_DIR, "flores200_dataset")


def load_sentences(split: str, src: str, limit: int = None):
    """Load FLORES source sentences (one per line)."""
    path = os.path.join(FLORES_DIR, f"{src}.{split}")
    if not os.path.exists(path):
        print(f"❌ FLORES file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]
    if limit:
        sents = sents[:limit]
    return sents


def load_reference(split: str, tgt: str = "eng_Latn", limit: int = None):
    """Load gold English references, if present."""
    path = os.path.join(FLORES_DIR, f"{tgt}.{split}")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        refs = [l.strip() for l in f if l.strip()]
    if limit:
        refs = refs[:limit]
    return refs


def is_translation_error(translation) -> bool:
    """Detect the placeholder strings the translator emits on failure."""
    if not translation:
        return True
    return translation.startswith("[Translation Error") or translation.startswith("[API Key")


def collect_sentence_stats(result) -> dict:
    """Extract per-sentence statistics from a PipelineResult."""
    per_token = []
    disambig_sources = Counter()
    missed_lemmas = Counter()       # lemma -> count for dict misses
    missed_forms = Counter()        # surface form -> count for dict misses
    fallback_forms = Counter()      # surface form -> count for fallback disambig
    total_candidates = 0
    ambiguous_tokens = 0            # >1 candidate reading
    n_tokens = 0
    n_punct = 0
    n_dict_hits = 0
    n_empty_lemma = 0

    upos_by_id = {t.id: t.upos for t in result.parsed_sentence.tokens}

    for morph, entry, tok in zip(result.morph_tokens, result.dict_entries,
                                 result.analysis.tokens):
        n_tokens += 1
        is_punct = (morph.pos == "Punctuation")
        if is_punct:
            n_punct += 1

        disambig_sources[morph.disambig_source] += 1
        if morph.disambig_source == "fallback" and not is_punct:
            fallback_forms[morph.form] += 1
        total_candidates += morph.total_candidates
        if morph.total_candidates > 1:
            ambiguous_tokens += 1
        if not morph.lemma:
            n_empty_lemma += 1

        has_dict = entry is not None
        if has_dict:
            n_dict_hits += 1
        elif not is_punct:
            missed_lemmas[morph.lemma] += 1
            missed_forms[morph.form] += 1

        per_token.append({
            "id": tok.id,
            "form": tok.form,
            "lemma": tok.lemma,
            "uly": tok.uly,
            "pos": tok.pos,
            "upos": upos_by_id.get(tok.id, "_"),
            "features": list(morph.features),
            "deprel": tok.deprel,
            "head_id": tok.head_id,
            "disambig_source": morph.disambig_source,
            "n_candidates": morph.total_candidates,
            "has_dict": has_dict,
            "gloss": tok.dict_gloss if has_dict else None,
            "near": [
                {"headword": e.headword_uey, "uly": e.headword_uly, "gloss": e.short_gloss}
                for e in tok.near_entries
            ],
        })

    n_content = n_tokens - n_punct
    return {
        "sentence": result.sentence,
        "translation": result.translation,
        "n_tokens": n_tokens,
        "n_punct": n_punct,
        "n_dict_hits": n_dict_hits,
        "dict_coverage": (n_dict_hits / n_content) if n_content else None,
        "n_ambiguous": ambiguous_tokens,
        "n_empty_lemma": n_empty_lemma,
        "per_token": per_token,
        # counters kept per-sentence for potential re-aggregation
        "_disambig_sources": dict(disambig_sources),
        "_missed_lemmas": dict(missed_lemmas),
        "_missed_forms": dict(missed_forms),
        "_fallback_forms": dict(fallback_forms),
    }


def aggregate_stats(sent_stats, total_time, translated) -> dict:
    """Aggregate per-sentence statistics into a run-level report."""
    n_sents = len(sent_stats)
    agg = Counter()
    missed_lemmas = Counter()
    missed_forms = Counter()
    fallback_forms = Counter()
    pos_dist = Counter()
    deprel_dist = Counter()
    candidate_dist = Counter()

    n_content_tokens = 0
    n_dict_hits_total = 0
    n_ambiguous_total = 0
    n_empty_lemma_total = 0
    cov_per_sent = []

    for s in sent_stats:
        n_content = s["n_tokens"] - s["n_punct"]
        n_content_tokens += n_content
        n_dict_hits_total += s["n_dict_hits"]
        n_ambiguous_total += s["n_ambiguous"]
        n_empty_lemma_total += s["n_empty_lemma"]
        if s["dict_coverage"] is not None:
            cov_per_sent.append((s["dict_coverage"], s["sentence"]))
        for k, v in s["_disambig_sources"].items():
            agg[k] += v
        for k, v in s["_missed_lemmas"].items():
            missed_lemmas[k] += v
        for k, v in s["_missed_forms"].items():
            missed_forms[k] += v
        for k, v in s["_fallback_forms"].items():
            fallback_forms[k] += v
        for t in s["per_token"]:
            pos_dist[t["pos"]] += 1
            deprel_dist[t["deprel"]] += 1
            candidate_dist[t["n_candidates"]] += 1

    cov_per_sent.sort()
    n_trans_errors = sum(
        1 for s in sent_stats if translated and is_translation_error(s["translation"])
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 1),
        "sec_per_sentence": round(total_time / n_sents, 3) if n_sents else None,
        "n_sentences": n_sents,
        "n_tokens": sum(s["n_tokens"] for s in sent_stats),
        "n_punct_tokens": sum(s["n_punct"] for s in sent_stats),
        "n_content_tokens": n_content_tokens,
        "n_empty_lemma_tokens": n_empty_lemma_total,

        # --- dictionary coverage ---
        "dict_hits": n_dict_hits_total,
        "dict_coverage": round(n_dict_hits_total / n_content_tokens, 4) if n_content_tokens else None,
        "n_unique_missed_lemmas": len(missed_lemmas),
        "top_missed_lemmas": [{"lemma": l, "count": c} for l, c in missed_lemmas.most_common(50)],
        "top_missed_forms": [{"form": f, "count": c} for f, c in missed_forms.most_common(50)],
        "worst_coverage_sentences": [
            {"coverage": round(c, 3), "sentence": sent[:120]}
            for c, sent in cov_per_sent[:10]
        ],

        # --- disambiguation ---
        "disambig_source_counts": dict(agg),
        "disambig_source_fractions": {
            k: round(v / sum(agg.values()), 4) for k, v in agg.items()
        } if agg else {},
        "fallback_forms": [{"form": f, "count": c} for f, c in fallback_forms.most_common(50)],
        "n_ambiguous_tokens": n_ambiguous_total,
        "ambiguous_token_fraction": round(n_ambiguous_total / n_content_tokens, 4) if n_content_tokens else None,
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(candidate_dist.items())
        },

        # --- tag distributions ---
        "pos_distribution": dict(pos_dist.most_common()),
        "deprel_distribution": dict(deprel_dist.most_common()),

        # --- translation ---
        "translated": translated,
        "n_translation_errors": n_trans_errors if translated else None,
        "translation_error_rate": round(n_trans_errors / n_sents, 4) if translated and n_sents else None,
    }


def render_summary(stats: dict) -> str:
    """Render the run summary as text; printed to console and persisted as
    <split>.stats.md so later tooling can read it without re-running."""
    lines = []
    ap = lines.append
    ap("\n" + "=" * 70)
    ap("📊 FLORES DEV RUN — PIPELINE STATISTICS")
    ap("=" * 70)
    ap(f"Sentences: {stats['n_sentences']}   Tokens: {stats['n_tokens']} "
       f"(content: {stats['n_content_tokens']})   "
       f"Time: {stats['total_time_sec']}s ({stats['sec_per_sentence']}s/sent)")
    ap("-" * 70)
    ap(f"📚 Dictionary coverage:  {stats['dict_hits']}/{stats['n_content_tokens']} "
       f"= {stats['dict_coverage']:.1%}")
    ap(f"   Unique missed lemmas: {stats['n_unique_missed_lemmas']}")
    ap(f"   Empty-lemma tokens:   {stats['n_empty_lemma_tokens']}")
    ap("-" * 70)
    ap("🔀 Disambiguation sources:")
    for k, v in sorted(stats["disambig_source_counts"].items()):
        ap(f"   {k:15s} {v:6d}  ({stats['disambig_source_fractions'][k]:.1%})")
    ap(f"   Ambiguous tokens (>1 reading): {stats['n_ambiguous_tokens']} "
       f"({stats['ambiguous_token_fraction']:.1%})")
    ap("-" * 70)
    ap("🏆 Top 15 missed lemmas (extend lookup variants / dictionary):")
    for item in stats["top_missed_lemmas"][:15]:
        ap(f"   {item['lemma']:25s} x{item['count']}")
    if stats["fallback_forms"]:
        ap("⚠️  Top fallback (failed-alignment) forms:")
        for item in stats["fallback_forms"][:10]:
            ap(f"   {item['form']:25s} x{item['count']}")
    if stats["translated"]:
        ap("-" * 70)
        ap(f"🌐 Translation errors: {stats['n_translation_errors']} "
           f"({stats['translation_error_rate']:.1%})")
    ap("=" * 70)
    return "\n".join(lines)


def print_summary(stats: dict):
    print(render_summary(stats))


def main():
    parser = argparse.ArgumentParser(
        description="Run a FLORES-200 split through the Uyghur linguistic "
                    "pipeline and collect inspection statistics (see module docstring)."
    )
    parser.add_argument("--split", type=str, default="dev",
                        help="FLORES split to evaluate (e.g. 'dev')")
    parser.add_argument("--src", type=str, default="uig_Arab",
                        help="FLORES source language code (file prefix)")

    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to the first N sentences (quick smoke test)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(BASE_DIR, "stats", "flores_dev"),
                        help="Directory for results and statistics")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Sentences per pipeline batch")

    parser.add_argument("--translate", action=argparse.BooleanOptionalAction, default=False,
                        help="Execute downstream LLM translation (ref: --no-translate)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI-compatible server base URL (overrides env OPENROUTER_BASE_URL)")
    parser.add_argument("--model", type=str, default=None,
                        help="Downstream LLM model identifier (overrides env TRANSLATION_LLM_MODEL)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max LLM output tokens (None = unlimited, for reasoning models)")

    args = parser.parse_args()

    # Build Configuration
    config = PipelineConfig()
    if args.translate:
        if args.base_url:
            config.llm_api_base = args.base_url
        if args.model:
            config.llm_model = args.model
        config.llm_max_tokens = args.max_tokens  # None = unlimited (reasoning models)
        print(f"🔌 Auto-detected model at {config.llm_api_base}: {config.llm_model}")
    else:
        print("🔒 Translation disabled — analysis/statistics only.")

    sentences = load_sentences(args.split, args.src, args.limit)
    print(f"📖 Loaded {len(sentences)} sentences from {args.src}.{args.split}")

    pipeline = UyghurLinguisticPipeline(config=config, load_models=True)
    print("✅ Pipeline ready!\n")

    # Process in batches
    sent_stats = []
    start = time.time()
    n_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(sentences), args.batch_size),
                  total=n_batches, desc=f"Processing {args.split}"):
        batch = sentences[i:i + args.batch_size]
        results = pipeline.process_batch(batch, translate=args.translate)
        for r in results:
            sent_stats.append(collect_sentence_stats(r))

    total_time = time.time() - start
    stats = aggregate_stats(sent_stats, total_time, args.translate)

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.split}.results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for s in sent_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats_path = os.path.join(args.output_dir, f"{args.split}.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Results: {results_path}\n              Stats: {stats_path}")
    print_summary(stats)

    # Persist the console summary so tooling can read it without re-running.
    summary_path = os.path.join(args.output_dir, f"{args.split}.stats.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_summary(stats) + "\n")


if __name__ == "__main__":
    main()
