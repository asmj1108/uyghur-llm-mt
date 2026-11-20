"""
score_exp5.py - Exp 5 analysis: thinking mode x C3L.

Per model, the 2x2 grid (REVISED — S arm dropped; see cells.py):
  arm A  greedy    thinking OFF, t=0 (reuse)
  arm T  thinkrec  thinking ON + recommended sampling

Contrasts (paired bootstrap, chrF++):
  T.C3L − A.C3L    thinking (+decoding) effect on the lean optimum
  T.C0  − A.C0     thinking effect on the zero-shot baseline
  ΔDID = [T.C3L−T.C0] − [A.C3L−A.C0]   does thinking change what the gloss
                                        column is worth? (the headline)

NOTE: T changes thinking AND sampling vs A (Exp2b's main-arm convention; the
offrec control was dropped — thinking-off sampling measured inert, and
thinking+greedy is measured-catastrophic for Qwen).

Mechanistic analysis (arm T only, unique to this experiment):
  gloss-citation rate — do the pushed glosses get reasoned about? For each
  sentence, citation targets = the first-sense keyword + Latin lemma of every
  dictionary-covered content token; an episode "cites" if any target appears
  as a word in its reasoning text. C0-T gives the baseline (same targets,
  never shown in the prompt): the signal is the C3L-T vs C0-T gap.

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" -u experiments/exp5_thinking_static/score_exp5.py
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root (two levels up): running this script directly puts only its own
# directory on sys.path, so the `experiments` package would not resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from experiments.exp5_thinking_static.cells import (  # noqa: E402
    FLORES_RESULTS_PATH, MODELS, RESULTS_DIR, cell_path, thinking_stats,
)

ARMS = ["greedy", "thinkrec"]
CONDS = {"c0_baseline": "C0", "c3_dict_lemma": "C3L"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "exp5_metrics.json"))
    ap.add_argument("--summary", default=os.path.join(RESULTS_DIR, "exp5_summary.md"))
    return ap.parse_args()


def load_run(path):
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
    return hyps, refs


def load_recs(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def boot_pair(stats_a, stats_b, n_boot, seed=12345):
    rng = np.random.default_rng(seed)
    n = len(stats_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (_corpus_from_stats(stats_b[idx])
                     - _corpus_from_stats(stats_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return (_corpus_from_stats(stats_b) - _corpus_from_stats(stats_a),
            lo, hi, p)


def boot_did(stats_c0_s, stats_c3l_s, stats_c0_t, stats_c3l_t, n_boot,
             seed=12345):
    """Bootstrap of [T.C3L−T.C0] − [A.C3L−A.C0] (the interaction)."""
    rng = np.random.default_rng(seed)
    n = len(stats_c0_s)

    def did(idx):
        return ((_corpus_from_stats(stats_c3l_t[idx])
                 - _corpus_from_stats(stats_c0_t[idx]))
                - (_corpus_from_stats(stats_c3l_s[idx])
                   - _corpus_from_stats(stats_c0_s[idx])))
    point = did(np.arange(n))
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        deltas[i] = did(rng.integers(0, n, n))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return point, lo, hi, p


# ---------------------------------------------------------------------------
# Gloss-citation analysis (arm T)
# ---------------------------------------------------------------------------
def citation_targets(record: dict) -> List[str]:
    """First-sense keyword + Latin lemma per dictionary-covered content token
    (the material C3L pushes into the prompt)."""
    targets = []
    for t in record["per_token"]:
        if t["pos"] == "Punctuation" or not t.get("has_dict"):
            continue
        gloss = t.get("gloss") or ""
        senses = gloss.split(": ", 1)[1] if ": " in gloss else gloss
        first = re.split(r"/\s*\d+\.|;|,", senses)[0]
        first = re.sub(r"\[.*?\]", "", first).strip(" .")
        if first:
            kw = first.split()[0].lower()
            if len(kw) >= 3 and kw.isalpha():
                targets.append(kw)
        uly = (t.get("uly") or "").strip()
        if len(uly) >= 3 and uly.isalpha():
            targets.append(uly.lower())
    return targets


def citation_stats(recs: Dict[int, dict], records: List[dict]) -> Dict:
    """How often the reasoning text mentions the pushed gloss material."""
    n = cited = 0
    distinct = []
    in_translation = 0
    for sent_idx, rec in recs.items():
        reasoning = (rec.get("reasoning") or "").strip().lower()
        translation = (rec.get("translation") or "").strip().lower()
        if not reasoning:
            continue
        n += 1
        rwords = set(re.findall(r"[a-zà-öø-ÿ'-]+", reasoning))
        twords = set(re.findall(r"[a-zà-öø-ÿ'-]+", translation))
        hits = {tg for tg in citation_targets(records[sent_idx])
                if tg in rwords}
        if hits:
            cited += 1
            distinct.append(len(hits))
            in_translation += bool(hits & twords)
    return {
        "episodes_with_reasoning": n,
        "citation_rate": round(cited / n, 3) if n else None,
        "distinct_citations_mean": (round(float(np.mean(distinct)), 1)
                                    if distinct else 0.0),
        "cited_and_in_translation": (round(in_translation / cited, 3)
                                     if cited else None),
    }


def main():
    args = parse_args()
    chrf = CHRF(word_order=2)
    records = [json.loads(l) for l in open(FLORES_RESULTS_PATH,
                                           encoding="utf-8")]

    # ---- load the grid, skipping models with missing/partial cells --------
    runs = {}   # (name, cond, arm) -> entry
    scored = []
    for model_id, tag, name in MODELS:
        cells = {(c, a): cell_path(tag, c, a) for c in CONDS for a in ARMS}
        missing = [p for p in cells.values() if not os.path.exists(p)]
        if missing:
            print(f"[exp5-score] SKIP {name}: missing {len(missing)} cells "
                  f"(e.g. {os.path.basename(missing[0])})", flush=True)
            continue
        lens = {k: len(load_run(p)[0]) for k, p in cells.items()}
        if len(set(lens.values())) != 1:
            print(f"[exp5-score] SKIP {name}: unequal cell sizes "
                  f"{set(lens.values())}", flush=True)
            continue
        for (c, a), path in cells.items():
            hyps, refs = load_run(path)
            entry = {
                "hyps": hyps, "refs": refs,
                "stats": np.array([_segment_stats(h, r)
                                   for h, r in zip(hyps, refs)]),
            }
            if a == "thinkrec":
                entry["recs"] = load_recs(path)
            runs[(name, CONDS[c], a)] = entry
        scored.append(name)
        print(f"[exp5-score] loaded {name}: 2 conds x {len(ARMS)} arms "
              f"(n={list(lens.values())[0]})", flush=True)
    if not scored:
        sys.exit("[exp5-score] no complete model grids found — nothing to do")

    # ---- quality (+ COMET unless disabled) ---------------------------------
    scores, comets = {}, {}
    if not args.no_comet:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(ckpt)
    for key, entry in runs.items():
        scores[key] = chrf.corpus_score(entry["hyps"],
                                        [entry["refs"]]).score
        if not args.no_comet:
            data = [{"src": records[i]["sentence"], "mt": h, "ref": r}
                    for i, (h, r) in enumerate(zip(entry["hyps"],
                                                    entry["refs"]))]
            comets[key] = float(comet_model.predict(
                data, batch_size=args.comet_batch, gpus=1,
                progress_bar=False).system_score)
        print(f"[exp5-score] {key[0]} {key[1]} {key[2]}: "
              f"chrF++ {scores[key]:.2f}", flush=True)
    if not args.no_comet:
        del comet_model
        import torch
        torch.cuda.empty_cache()

    # ---- contrasts -----------------------------------------------------------
    deltas = {}
    for name in scored:
        for cond in ("C0", "C3L"):
            a = runs[(name, cond, "greedy")]["stats"]
            t = runs[(name, cond, "thinkrec")]["stats"]
            d, lo, hi, p = boot_pair(a, t, args.n_boot)
            label = f"T.{cond} − A.{cond} (thinking)"
            deltas[(name, label)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[exp5-score] {name} {label}: {d:+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}] p={p:.4f}", flush=True)
        d, lo, hi, p = boot_did(
            runs[(name, "C0", "greedy")]["stats"],
            runs[(name, "C3L", "greedy")]["stats"],
            runs[(name, "C0", "thinkrec")]["stats"],
            runs[(name, "C3L", "thinkrec")]["stats"], args.n_boot)
        deltas[(name, "DID")] = {"delta": d, "lo": lo, "hi": hi, "p": p}
        print(f"[exp5-score] {name} DID [T.C3L−T.C0]−[A.C3L−A.C0]: {d:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] p={p:.4f}", flush=True)

    # ---- thinking behavior + gloss citations ----------------------------------
    tstats, cites = {}, {}
    for model_id, tag, name in MODELS:
        if name not in scored:
            continue
        for c in CONDS:
            path = cell_path(tag, c, "thinkrec")
            tstats[f"{name}|{CONDS[c]}"] = thinking_stats(path)
            if "recs" in runs[(name, CONDS[c], "thinkrec")]:
                cites[f"{name}|{CONDS[c]}"] = citation_stats(
                    runs[(name, CONDS[c], "thinkrec")]["recs"], records)
        # non-thinking arms: degeneration guard (looping finals)
        st = thinking_stats(cell_path(tag, "c3_dict_lemma", "greedy"))
        tstats[f"{name}|C3L|greedy"] = st

    # ---- persist ---------------------------------------------------------------
    out = {
        "models_scored": scored,
        "scores": {f"{k[0]}|{k[1]}|{k[2]}":
                   {"chrfpp": round(v, 2),
                    **({"comet": round(comets[k], 4)} if k in comets else {})}
                   for k, v in scores.items()},
        "contrasts": {f"{n}|{lab}": {"delta": round(v["delta"], 2),
                                      "ci_lo": round(v["lo"], 2),
                                      "ci_hi": round(v["hi"], 2),
                                      "p": round(v["p"], 4)}
                      for (n, lab), v in deltas.items()},
        "thinking_stats": tstats,
        "gloss_citations": cites,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown -----------------------------------------------------------------
    L = ["", "## Experiment 5 — thinking mode x C3L", ""]
    L += ["| Model | Cond | arm A greedy | arm T thinkrec |",
          "|" + "---|" * 4]
    for name in scored:
        for cond in ("C0", "C3L"):
            row = [name, cond]
            for a in ARMS:
                s = scores[(name, cond, a)]
                cell = f"{s:.2f}"
                if (name, cond, a) in comets:
                    cell += f" ({comets[(name, cond, a)]:.4f})"
                row.append(cell)
            L.append("| " + " | ".join(row) + " |")
    L += ["", "### Contrasts (paired bootstrap, chrF++)", "",
          "| Model | Contrast | Δ [95% CI] | p |", "|" + "---|" * 4]
    for (n, lab), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        L.append(f"| {n} | {lab} | {v['delta']:+.2f} "
                 f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")
    L += ["", "### Thinking behavior (arm T)", "",
          "| Model | Cond | answered | len-capped | loops (reason/answer) | "
          "reasoning chars mean/max |", "|" + "---|" * 6]
    for name in scored:
        for cond in ("C0", "C3L"):
            st = tstats[f"{name}|{cond}"]
            L.append(f"| {name} | {cond} | "
                     f"{100 * (st['answered_rate'] or 0):.1f}% "
                     f"| {st['length_capped']} "
                     f"| {st['looping_reasoning']}/{st['looping_answer']} "
                     f"| {st['reasoning_chars_mean']}/{st['reasoning_chars_max']} |")
    L += ["", "### Gloss citations in reasoning (arm T; targets = first-sense "
          "keyword + Latin lemma of every dict-covered token)", "",
          "| Model | Cond | eps w/ reasoning | citation rate | distinct cites "
          "| cited & in final |", "|" + "---|" * 6]
    for name in scored:
        for cond in ("C0", "C3L"):
            c = cites.get(f"{name}|{cond}")
            if not c:
                continue
            L.append(f"| {name} | {cond} | {c['episodes_with_reasoning']} "
                     f"| {c['citation_rate']} | {c['distinct_citations_mean']} "
                     f"| {c['cited_and_in_translation']} |")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp5-score] wrote {args.out} + {args.summary}")


if __name__ == "__main__":
    main()
