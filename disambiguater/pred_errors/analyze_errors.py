import os
import json
import random as _random
from itertools import combinations
import numpy as np

# =============================================================================
# CONFIG (Point PREDICTIONS to generated JSON files)
# =============================================================================
TEST_FILE = "dev.jsonl"
OUT_DIR = "error_analysis"
SEED = 42

PREDICTIONS = {
    "byt5-base":  "byt5-base_dev_preds.json",
    "qwen-0.8b":  "qwen-0.8b_dev_preds.json",
    "qwen-2b":    "qwen-2b_dev_preds.json",
    "xlmr-large": "xlmr_dev_preds.json"
}

KNOWN_SOURCES = ("llm_based", "rule_based", "dedup")
HONEST_SRC = "llm_based"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def fixed_perms(rows, seed=SEED):
    """Reproduce the eval-time fixed shuffle to reconstruct what the models saw."""
    rng = _random.Random(seed)
    perms = []
    for r in rows:
        perm = list(range(len(r.get("candidates", []))))
        rng.shuffle(perm)
        perms.append(perm)
    return perms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_jsonl(TEST_FILE)
    perms = fixed_perms(rows)

    # 1. Map true gold index to relative index in permutation
    gold_rel = []
    for r, p in zip(rows, perms):
        g = int(r.get("label_id", -1))
        gold_rel.append(p.index(g) if g in p else g)

    sources = [r.get("source", "unknown") for r in rows]

    # 2. Collect per-model predicted rel idx from JSONs
    model_pred = {}
    for name, path in PREDICTIONS.items():
        if not os.path.exists(path):
            print(f"⚠️ Warning: Missing predictions for {name} ({path})")
            continue
        with open(path, "r") as f:
            d = json.load(f)
            # Ensure it maps properly by integer row index
            model_pred[name] = [d.get(str(i)) for i in range(len(rows))]

    names = list(model_pred.keys())
    if not names:
        print("No prediction files found. Exiting.")
        return

    # ----- correctness matrix (restricted to honest llm_based slice) -----
    idx_llm = [i for i, s in enumerate(sources) if s == HONEST_SRC and gold_rel[i] >= 0]
    hit = {n: {i: (model_pred[n][i] == gold_rel[i]) for i in idx_llm} for n in names}

    # per-model accuracy (overall + per source)
    acc = {}
    for n in names:
        acc[n] = {}
        for src in ("all",) + KNOWN_SOURCES:
            ids = [i for i in range(len(rows)) if gold_rel[i] >= 0
                   and (src == "all" or sources[i] == src)]
            if ids:
                acc[n][src] = np.mean([model_pred[n][i] == gold_rel[i] for i in ids])

    # ----- pairwise error overlap (Jaccard on the SET of wrong rows) -----
    err = {n: {i for i in idx_llm if not hit[n][i]} for n in names}
    pairwise = {}
    for a, b in combinations(names, 2):
        inter = len(err[a] & err[b])
        union = len(err[a] | err[b]) or 1
        pairwise[f"{a}|{b}"] = {
            "jaccard": inter / union,
            "shared_errors": inter,
            f"{a}_errors": len(err[a]),
            f"{b}_errors": len(err[b]),
        }

    # ----- all-wrong: observed vs. expected under independence -----
    n_llm = len(idx_llm)
    err_rates = {n: len(err[n]) / n_llm for n in names}
    all_wrong = [i for i in idx_llm if all(not hit[n][i] for n in names)]

    # Calculate Expected under independence
    expected_all_wrong = n_llm * np.prod([err_rates[n] for n in names])

    # ----- dump shared-failure rows for manual inspection -----
    dump = []
    for i in all_wrong:
        r = rows[i]
        cands = [r["candidates"][j] for j in perms[i]]
        dump.append({
            "row": i,
            "sentence": r["sentence"],
            "target_word": r["target_word"],
            "gold_idx": gold_rel[i] + 1,
            "gold_candidate": cands[gold_rel[i]],
            "candidates": cands,
            "model_preds": {n: (model_pred[n][i] + 1 if model_pred[n][i] is not None else None)
                            for n in names},
            "pseudo_gold_reasoning": r.get("reasoning", ""),
        })

    report = {
        "n_llm_based": n_llm,
        "per_model_accuracy": acc,
        "per_model_error_rate_llm": err_rates,
        "pairwise_error_overlap": pairwise,
        "all_models_wrong": {
            "observed": len(all_wrong),
            "expected_if_independent": round(float(expected_all_wrong), 2),
            "over_representation_factor": round(len(all_wrong) / (expected_all_wrong or 1e-9), 2),
        },
    }

    json.dump(report, open(f"{OUT_DIR}/report.json", "w"), indent=2, ensure_ascii=False)
    json.dump(dump, open(f"{OUT_DIR}/shared_failures.json", "w"), indent=2, ensure_ascii=False)

    # ----- console summary -----
    print("\n" + "=" * 60)
    print("  CROSS-MODEL ERROR ANALYSIS (llm_based slice)")
    print("=" * 60)
    print(f"n = {n_llm}")
    for n in names:
        print(f"  {n:12s} acc={acc[n]['all']:.4f}  err={err_rates[n]:.4f}")
    print("\n  Pairwise error Jaccard:")
    for k, v in pairwise.items():
        print(f"    {k:28s} J={v['jaccard']:.3f}  shared={v['shared_errors']}")
    print("\n  ALL models wrong on the same row:")
    print(f"    observed = {len(all_wrong)}")
    print(f"    expected (independent) = {expected_all_wrong:.1f}")
    print(f"    over-representation = {report['all_models_wrong']['over_representation_factor']}x")
    print(f"\n  -> {len(all_wrong)} shared failures dumped to {OUT_DIR}/shared_failures.json\n")


if __name__ == "__main__":
    main()