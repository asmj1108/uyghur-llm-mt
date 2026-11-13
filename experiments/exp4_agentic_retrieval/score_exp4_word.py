"""
score_exp4_word.py - Exp 4 follow-up analysis: word-level get_syntax (E4-W)
vs sentence-level syntax at a raised budget (E4-A2).

Cells: C0 / C3L / E4-A (all reused, on disk) + E4-A2 / E4-W (new runs).
Contrasts (paired bootstrap, chrF++):
  E4-W − E4-A2   granularity (identical budget, only get_syntax addressing)
  E4-A2 − E4-A   budget (fixed 24 calls -> proportional α=2, floor 8, 64/12)
  E4-W − C3L     word-syntax pull vs best static push
  E4-A2 − C3L    budget-raised pull vs best static push
  E4-W − E4-A    combined granularity+budget effect

Behavior: standard agent table + syntax adoption (calls/ep, episodes with
>=1 syntax call, resolution mix) + EDGE-COVERAGE analysis (E4-W):
a call on token t reveals arc(t->head(t)) and arcs(d->t) for direct
dependents d; the head's deprel counts only when the head is root (endpoint
inferable). coverage = |revealed arcs| / |arcs|; mode bins
none / spot (<= 1/3) / partial / assembly (> 3/4); per-sentence chrF by bin
is the H4 test (assembly-heavy episodes worse than spot-checks).

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" -u experiments/exp4_agentic_retrieval/score_exp4_word.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "results")
E3 = os.path.join(HERE, "..", "exp3_cross_family", "results")
# Qwen's C0 baseline lives in Exp 1 (different naming pattern, too)
E1 = os.path.join(HERE, "..", "exp1_component_ablation", "results")
FLORES_ANALYSIS = os.path.join(HERE, "..", "..", "stats", "flores_dev", "dev.results.jsonl")

# Ministral-3-8B was dropped mid-run (too slow; E4-W killed at 128/997,
# E4-A2 completed) — re-add the entry below once E4-W finishes to score it.
# The loop below skips any model with missing/partial cells anyway.
MODELS = [
    ("qwen3.5-9b", "Qwen3.5-9B",
     f"{E1}/c0_baseline_qwen3.5-9b.results.jsonl",
     f"{E3}/qwen3.5-9b_c3_dict_lemma.results.jsonl"),
    ("gemma-4-e4b-it", "Gemma-4-E4B",
     f"{E3}/gemma-4-e4b-it_c0_baseline.results.jsonl",
     f"{E3}/gemma-4-e4b-it_c3_dict_lemma.results.jsonl"),
    # ("ministral-3-8b-instruct-2512-bf16", "Ministral-3-8B",
    #  f"{E3}/ministral-3-8b-instruct-2512-bf16_c0_baseline.results.jsonl",
    #  f"{E3}/ministral-3-8b-instruct-2512-bf16_c3_dict_lemma.results.jsonl"),
]

CONDS = ["C0", "C3L", "E4-A", "E4-A2", "E4-W"]

CONTRASTS = [
    ("C3L", "E4-W", "E4-W − C3L (word-syntax pull vs best push)"),
    ("C3L", "E4-A2", "E4-A2 − C3L (budget-raised pull vs best push)"),
    ("E4-A", "E4-W", "E4-W − E4-A (granularity + budget)"),
    ("E4-A2", "E4-W", "E4-W − E4-A2 (word-level vs sentence-level syntax)"),
    ("E4-A", "E4-A2", "E4-A2 − E4-A (raised budget vs fixed 24 calls)"),
]

# bin edges for coverage modes (DESIGN_word_syntax.md §5)
SPOT, ASSEMBLY = 1.0 / 3.0, 3.0 / 4.0


def load_run(path):
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
    return hyps, refs


def load_recs(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def bootstrap_between(stats_a, stats_b, n_boot, seed=12345):
    rng = np.random.default_rng(seed)
    n = stats_a.shape[0]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (_corpus_from_stats(stats_b[idx])
                     - _corpus_from_stats(stats_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return (_corpus_from_stats(stats_b) - _corpus_from_stats(stats_a),
            lo, hi, p)


def behavior(recs):
    n = len(recs)
    calls = [r["n_calls"] for r in recs]
    rounds_ = [r["n_rounds"] for r in recs]
    fin, mix = Counter(), Counter()
    for r in recs:
        fin[r["finish"]] += 1
        mix.update(r["tool_mix"])
    return {
        "n": n,
        "episodes_with_calls": sum(c > 0 for c in calls),
        "calls_mean": round(float(np.mean(calls)), 1),
        "calls_p95": float(np.percentile(calls, 95)),
        "rounds_mean": round(float(np.mean(rounds_)), 1),
        "budget_forced": fin.get("budget_forced", 0),
        "round_cap_forced": fin.get("round_cap", 0),
        "context_overflow": fin.get("context_overflow", 0),
        "parse_fail": sum(r["n_parse_failures"] for r in recs),
        "skipped": sum(r["n_skipped_calls"] for r in recs),
        "unresolved": sum(r["n_unresolved"] for r in recs),
        "tool_mix": dict(mix),
        "tokens_in_mean": sum(r["tokens_in"] for r in recs) // n,
        "tokens_out_mean": sum(r["tokens_out"] for r in recs) // n,
    }


def syntax_adoption(recs):
    """get_syntax usage: calls, episodes engaged, resolution mix (E4-W)."""
    calls = [r["tool_mix"].get("get_syntax", 0) for r in recs]
    res = Counter()
    distinct = []
    for r in recs:
        seen = set()
        for c in r["call_log"]:
            if c.get("tool") == "get_syntax":
                # sentence-level get_syntax records no word resolution
                # (None) — label it n/a, not "unresolved"
                res[c.get("resolution") or "n/a (no word)"] += 1
                if c.get("resolved"):
                    seen.add(c["resolved"])
        if seen:
            distinct.append(len(seen))
    return {
        "syntax_calls": sum(calls),
        "episodes_with_syntax": sum(c > 0 for c in calls),
        "calls_per_engaged_ep": (round(float(np.mean(
            [c for c in calls if c > 0])), 1) if any(calls) else 0.0),
        "distinct_tokens_per_engaged_ep": (round(float(np.mean(distinct)), 1)
                                           if distinct else 0.0),
        "resolution": dict(res),
    }


def edge_coverage(recs, records):
    """Replay get_syntax calls per episode; returns per-episode dicts."""
    out = []
    for r in recs:
        toks = records[r["id"]]["per_token"]
        by_id = {t["id"]: t for t in toks}
        root_ids = {t["id"] for t in toks if t["head_id"] == 0}
        punct = {t["id"] for t in toks if t["pos"] == "Punctuation"}
        total = len(toks)  # one arc per token (root arc: token -> ROOT)
        revealed = set()
        for c in r["call_log"]:
            if c.get("tool") != "get_syntax":
                continue
            tid = c.get("resolved")
            if tid is None or tid not in by_id:
                continue
            revealed.add(tid)  # w -> head(w)
            for d in toks:    # d -> w for each direct dependent
                if d["head_id"] == tid and d["id"] != tid:
                    revealed.add(d["id"])
            h = by_id[tid]["head_id"]
            if h in root_ids:  # head's deprel = root -> endpoint inferable
                revealed.add(h)
        content_total = total - len(punct)
        out.append({
            "id": r["id"],
            "n_syntax_calls": r["tool_mix"].get("get_syntax", 0),
            "coverage": len(revealed) / total if total else 0.0,
            "coverage_content": (len(revealed - punct) / content_total
                                if content_total else 0.0),
        })
    return out


def coverage_bin(cov):
    if cov <= 0.0:
        return "none"
    if cov <= SPOT:
        return "spot"
    if cov <= ASSEMBLY:
        return "partial"
    return "assembly"


def main(n_boot=5000):
    chrf = CHRF(word_order=2)
    records = [json.loads(l) for l in open(FLORES_ANALYSIS, encoding="utf-8")]

    lines = ["", "## Experiment 4 follow-up — word-level get_syntax (E4-W) "
              "and budget raise (E4-A2)", ""]

    runs = {}  # (model_name, cond) -> dict
    scored_models = []  # (tag, name, c0_path, c3l_path) actually loaded
    for tag, name, c0_path, c3l_path in MODELS:
        cells = [("C0", c0_path), ("C3L", c3l_path),
                 ("E4-A", f"{E4}/{tag}_e4-a.results.jsonl"),
                 ("E4-A2", f"{E4}/{tag}_e4-a2.results.jsonl"),
                 ("E4-W", f"{E4}/{tag}_e4-w.results.jsonl")]
        missing = [p for _, p in cells if not os.path.exists(p)]
        if missing:
            print(f"[word-score] SKIP {name}: missing {missing}", flush=True)
            continue
        lens = {c: len(load_run(p)[0]) for c, p in cells}
        if len(set(lens.values())) != 1:
            # partial cells would silently truncate the paired bootstrap's
            # positional alignment (zip stops at the shortest file)
            print(f"[word-score] SKIP {name}: unequal cell sizes {lens}",
                  flush=True)
            continue
        for cond, path in cells:
            hyps, refs = load_run(path)
            stats = np.array([_segment_stats(h, r) for h, r in zip(hyps, refs)])
            entry = {"hyps": hyps, "refs": refs, "stats": stats}
            if cond.startswith("E4"):
                entry["recs"] = load_recs(path)
            runs[(name, cond)] = entry
        scored_models.append((tag, name, c0_path, c3l_path))
        print(f"[word-score] loaded {name}: C0/C3L/E4-A/E4-A2/E4-W "
              f"(n={lens['C0']})", flush=True)
    if not scored_models:
        sys.exit("[word-score] no complete model cells found — nothing to do")

    # ---- quality + COMET ------------------------------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    scores, comets = {}, {}
    for key, entry in runs.items():
        scores[key] = chrf.corpus_score(entry["hyps"], [entry["refs"]]).score
        data = [{"src": records[i]["sentence"], "mt": h, "ref": r}
                for i, (h, r) in enumerate(zip(entry["hyps"], entry["refs"]))]
        comets[key] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[word-score] {key[0]} {key[1]}: chrF++ {scores[key]:.2f} "
              f"COMET {comets[key]:.4f}", flush=True)
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- contrasts ---------------------------------------------------------------
    deltas = {}
    for _, name, _, _ in scored_models:
        for a, b, desc in CONTRASTS:
            d, lo, hi, p = bootstrap_between(runs[(name, a)]["stats"],
                                             runs[(name, b)]["stats"], n_boot)
            deltas[(name, a, b, desc)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[word-score] {name} {desc}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"p={p:.4f}", flush=True)

    # ---- behavior / adoption / coverage ----------------------------------------------
    beh, syn, cov_all = {}, {}, {}
    for _, name, _, _ in scored_models:
        for cond in ("E4-A", "E4-A2", "E4-W"):
            recs = runs[(name, cond)]["recs"]
            beh[f"{name}|{cond}"] = behavior(recs)
            syn[f"{name}|{cond}"] = syntax_adoption(recs)
        cov_all[name] = edge_coverage(runs[(name, "E4-W")]["recs"], records)

    # per-sentence chrF by coverage bin (E4-W, H4 test)
    bin_scores = {}
    for _, name, _, _ in scored_models:
        hyps = runs[(name, "E4-W")]["hyps"]
        refs = runs[(name, "E4-W")]["refs"]
        per_bin = {b: [] for b in ("none", "spot", "partial", "assembly")}
        for c in cov_all[name]:
            b = coverage_bin(c["coverage"])
            s = chrf.sentence_score(hyps[c["id"]], [refs[c["id"]]]).score
            per_bin[b].append(s)
        bin_scores[name] = {b: {"n": len(v),
                                 "chrf_mean": round(float(np.mean(v)), 2)
                                 if v else None}
                            for b, v in per_bin.items()}

    # ---- persist -----------------------------------------------------------------
    out = {
        "models_scored": [n for _, n, _, _ in scored_models],
        "scores": {f"{n}|{c}": {"chrfpp": round(scores[(n, c)], 2),
                                 "comet": round(comets[(n, c)], 4)}
                   for _, n, _, _ in scored_models for c in CONDS},
        "contrasts": {f"{n}|{b}−{a}": {"desc": d, "delta": round(v["delta"], 2),
                                        "ci_lo": round(v["lo"], 2),
                                        "ci_hi": round(v["hi"], 2),
                                        "p": round(v["p"], 4)}
                      for (n, a, b, d), v in deltas.items()},
        "behavior": beh, "syntax_adoption": syn,
        "coverage": {n: {"mean": round(float(np.mean([c["coverage"]
                                                       for c in cov])), 4),
                          "mean_content": round(float(np.mean(
                              [c["coverage_content"] for c in cov])), 4),
                          "zero_call_eps": sum(c["n_syntax_calls"] == 0
                                               for c in cov)}
                     for n, cov in cov_all.items()},
        "coverage_bin_chrf": bin_scores,
    }
    with open(os.path.join(E4, "exp4_word_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown --------------------------------------------------------------------
    lines += ["### Quality", "", "| Model | Cond | chrF++ | COMET |",
              "|" + "---|" * 4]
    for _, name, _, _ in scored_models:
        for c in CONDS:
            lines.append(f"| {name} | {c} | {scores[(name, c)]:.2f} "
                         f"| {comets[(name, c)]:.4f} |")
    lines += ["", "### Contrasts (paired bootstrap, chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |", "|" + "---|" * 4]
    for (n, a, b, d), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        lines.append(f"| {n} | {d} | {v['delta']:+.2f} "
                     f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")

    lines += ["", "### Agent behavior", "",
              "| Model | Cond | ep w/ calls | calls/ep | p95 | rounds/ep | "
              "budget-forced | skipped | unresolved | tok in/out |",
              "|" + "---|" * 10]
    for _, name, _, _ in scored_models:
        for cond in ("E4-A", "E4-A2", "E4-W"):
            b = beh[f"{name}|{cond}"]
            lines.append(f"| {name} | {cond} | {b['episodes_with_calls']}/{b['n']} "
                         f"| {b['calls_mean']} | {b['calls_p95']:.0f} "
                         f"| {b['rounds_mean']} | {b['budget_forced']} "
                         f"| {b['skipped']} | {b['unresolved']} "
                         f"| {b['tokens_in_mean']}/{b['tokens_out_mean']} |")

    lines += ["", "### get_syntax adoption", "",
              "| Model | Cond | syntax calls | eps ≥1 syntax | calls/engaged ep "
              "| distinct tokens/engaged ep | resolution |",
              "|" + "---|" * 7]
    for _, name, _, _ in scored_models:
        for cond in ("E4-A", "E4-A2", "E4-W"):
            s = syn[f"{name}|{cond}"]
            lines.append(f"| {name} | {cond} | {s['syntax_calls']} "
                         f"| {s['episodes_with_syntax']} "
                         f"| {s['calls_per_engaged_ep']} "
                         f"| {s['distinct_tokens_per_engaged_ep']} "
                         f"| {s['resolution']} |")

    lines += ["", "### Edge coverage (E4-W, arcs revealed / arcs)", "",
              "| Model | mean coverage | mean content-only | zero-call eps |",
              "|" + "---|" * 4]
    for _, name, _, _ in scored_models:
        c = out["coverage"][name]
        lines.append(f"| {name} | {c['mean']} | {c['mean_content']} "
                     f"| {c['zero_call_eps']} |")

    lines += ["", "Per-sentence chrF++ by coverage bin (H4: assembly < spot "
              "means partial structure beats the assembled picture)", "",
              "| Model | none | spot (≤⅓) | partial (⅓–¾] | assembly (>¾) |",
              "|" + "---|" * 5]
    for _, name, _, _ in scored_models:
        bs = bin_scores[name]
        fmt = lambda b: (f"{bs[b]['n']} eps, {bs[b]['chrf_mean']}"
                         if bs[b]["chrf_mean"] is not None else f"{bs[b]['n']} eps")
        lines.append(f"| {name} | {fmt('none')} | {fmt('spot')} "
                     f"| {fmt('partial')} | {fmt('assembly')} |")

    with open(os.path.join(E4, "exp4_word_summary.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[word-score] wrote exp4_word_metrics.json + exp4_word_summary.md "
          "+ ANALYSIS.md section")


if __name__ == "__main__":
    main()
