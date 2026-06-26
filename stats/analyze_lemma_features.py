import sys
import json
import re
from collections import Counter

INPUT_FILE = "uyghur_disambiguation_dataset.jsonl"

def extract_base_lemma(candidate_str):
    """
    Candidate looks like:
       'Lemma: قوغدا | Features: Verb, Transitive, ...'
    or (with copula):
       'Lemma: قوغداش | Features: Noun, Nominative + Combined with: Lemma: ئى | ...'
    We take only the FIRST 'Lemma:' value (treats +copula as same lemma).
    """
    m = re.search(r"Lemma:\s*(\S+)", candidate_str)
    return m.group(1) if m else None

def extract_feature_set(candidate_str):
    """
    Return the full glossed candidate string as the 'feature signature'.
    Two readings with identical lemma but different features will differ here.
    """
    return candidate_str.strip()

def analyze(input_file):
    total_words          = 0
    single_lemma_words   = 0   # all readings share ONE base lemma (feature-only ambiguity)
    multi_lemma_words    = 0   # readings span >1 base lemma
    distinct_lemma_dist  = Counter()   # how many distinct lemmas per word
    feature_signature_set = set()      # global pool of unique feature signatures (single-lemma cases)
    degenerate_words     = 0           # readings collapse to identical signature (shouldn't happen often)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            candidates = data.get("candidates", [])
            if len(candidates) <= 1:
                continue  # not ambiguous (rule fully resolved to 1 candidate doesn't store reduced list)

            total_words += 1

            base_lemmas = {extract_base_lemma(c) for c in candidates}
            base_lemmas.discard(None)

            distinct_lemma_dist[len(base_lemmas)] += 1

            if len(base_lemmas) == 1:
                single_lemma_words += 1
                # collect feature signatures for the closed-label feasibility check
                sigs = {extract_feature_set(c) for c in candidates}
                feature_signature_set.update(sigs)
                if len(sigs) < len(candidates):
                    degenerate_words += 1
            else:
                multi_lemma_words += 1

    # ==========================================
    # REPORT
    # ==========================================
    print("=" * 58)
    print("  LEMMA vs FEATURE AMBIGUITY ANALYSIS")
    print("=" * 58)
    print(f"Source file               : {input_file}")
    print(f"Total ambiguous words     : {total_words}")
    print("-" * 58)

    if total_words == 0:
        print("No ambiguous words found.")
        return

    sl_pct = single_lemma_words / total_words * 100
    ml_pct = multi_lemma_words  / total_words * 100

    print(f"Single-lemma (feature-only) : {single_lemma_words:>6}  ({sl_pct:.1f}%)")
    print(f"Multi-lemma  (lemma differs): {multi_lemma_words:>6}  ({ml_pct:.1f}%)")
    print("-" * 58)

    print("Distribution of distinct base lemmas per word:")
    for n in sorted(distinct_lemma_dist):
        print(f"   {n} distinct lemma(s): {distinct_lemma_dist[n]:>6} words")
    print("-" * 58)

    # print(f"Unique feature signatures (single-lemma pool): {len(feature_signature_set)}")
    # print(f"Degenerate words (identical signatures)      : {degenerate_words}")
    # print("=" * 58)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    analyze(path)