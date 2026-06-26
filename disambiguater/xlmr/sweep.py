#!/usr/bin/env python3
# =============================================================================
# Sweep driver for XLM-R Uyghur morphological disambiguation.
# Each run = isolated subprocess (clean CUDA + clean seed state).
# Strategy: tune ONE knob at a time, each over multiple seeds, report mean±std
# of the DEPLOYMENT metric (eval_cand_acc_llm_based).
# =============================================================================
import os
import sys
import json
import subprocess
import statistics
from datetime import datetime

TRAIN_SCRIPT = "train_xlmr.py"
SWEEP_ROOT   = "sweeps"
SEEDS        = [42, 1, 2, 3]
HEADLINE = "eval_cand_acc_llm_based"

os.makedirs(SWEEP_ROOT, exist_ok=True)

# -----------------------------------------------------------------------------
# Define experiments. Each entry is a name -> dict of CFG_* overrides.
# Start with a baseline, then change ONE knob per experiment.
# Comment out groups you are not running this session.
# -----------------------------------------------------------------------------
EXPERIMENTS = {}

# ---- Group 0: model size baseline (large vs base) ----
EXPERIMENTS["base_model_large"] = {"CFG_MODEL_NAME": "xlm-roberta-large"}
EXPERIMENTS["base_model_base"]  = {"CFG_MODEL_NAME": "xlm-roberta-base"}

# ---- Group 1: learning rate (tune first, biggest effect) ----
EXPERIMENTS["lr_7e6"] = {"CFG_LEARNING_RATE": "7e-6"}
EXPERIMENTS["lr_1e5"] = {"CFG_LEARNING_RATE": "1e-5"}
EXPERIMENTS["lr_2e5"] = {"CFG_LEARNING_RATE": "2e-5"}

# ---- Group 2: LLRD decay (run after fixing best LR) ----
# EXPERIMENTS["llrd_090"] = {"CFG_LLRD_DECAY": "0.9"}
# EXPERIMENTS["llrd_095"] = {"CFG_LLRD_DECAY": "0.95"}
# EXPERIMENTS["llrd_off"] = {"CFG_USE_LLRD": "false"}

# ---- Group 3: loss (run after fixing LR + LLRD) ----
# EXPERIMENTS["loss_focal"]    = {"CFG_LOSS_TYPE": "focal", "CFG_FOCAL_USE_POS_WEIGHT": "false"}
# EXPERIMENTS["loss_focal_pw"] = {"CFG_LOSS_TYPE": "focal", "CFG_FOCAL_USE_POS_WEIGHT": "true"}
# EXPERIMENTS["loss_weighted"] = {"CFG_LOSS_TYPE": "weighted"}
# EXPERIMENTS["loss_bce"]      = {"CFG_LOSS_TYPE": "bce"}

# ---- Group 4: batch size (gradient-noise regularization) ----
# EXPERIMENTS["bs_16"] = {"CFG_TRAIN_BATCH_SIZE": "16"}
# EXPERIMENTS["bs_32"] = {"CFG_TRAIN_BATCH_SIZE": "32"}


def run_one(exp_name, overrides, seed):
    run_name = f"{exp_name}__seed{seed}"
    out_dir  = os.path.join(SWEEP_ROOT, run_name)
    metrics_path = os.path.join(out_dir, "run_metrics.json")

    # resume: skip if already finished
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)

    env = os.environ.copy()
    env.update(overrides)
    env["CFG_SEED"]       = str(seed)
    env["CFG_OUTPUT_DIR"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "train.log")
    print(f"  ▶ {run_name}  overrides={overrides}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0 or not os.path.exists(metrics_path):
        print(f"  ✖ FAILED {run_name} (see {log_path})")
        return None
    with open(metrics_path) as f:
        return json.load(f)


def main():
    summary = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_name, overrides in EXPERIMENTS.items():
        print(f"\n=== Experiment: {exp_name} ===")
        scores = []
        per_seed = {}
        for seed in SEEDS:
            metrics = run_one(exp_name, overrides, seed)
            if metrics is None:
                continue
            val = metrics.get(HEADLINE)
            if val is None:
                print(f"  ⚠ {HEADLINE} missing in metrics for seed {seed}")
                continue
            scores.append(val)
            per_seed[seed] = {
                HEADLINE: val,
                "eval_cand_accuracy":       metrics.get("eval_cand_accuracy"),
                "eval_cand_acc_rule_based": metrics.get("eval_cand_acc_rule_based"),
                "eval_cand_acc_dedup":      metrics.get("eval_cand_acc_dedup"),
                "test_cand_acc_llm_based":  metrics.get("test_cand_acc_llm_based"),
                "test_cand_accuracy":     metrics.get("test_cand_accuracy"),
                "test_cand_acc_rule_based": metrics.get("test_cand_acc_rule_based"),
                "test_cand_acc_dedup":      metrics.get("test_cand_acc_dedup"),
            }
        if scores:
            mean = statistics.mean(scores)
            std  = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            print(f"  → {HEADLINE}: {mean:.4f} ± {std:.4f}  (n={len(scores)})")
            summary.append({
                "experiment": exp_name, "overrides": overrides,
                "mean": mean, "std": std, "n": len(scores),
                "scores": scores, "per_seed": per_seed,
            })

    # rank by mean deployment metric
    summary.sort(key=lambda x: x["mean"], reverse=True)
    out_file = os.path.join(SWEEP_ROOT, f"summary_{stamp}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  SWEEP SUMMARY (ranked by {HEADLINE})")
    print("=" * 64)
    for s in summary:
        print(f"  {s['experiment']:22s} {s['mean']:.4f} ± {s['std']:.4f}"
              f"  (n={s['n']})  {s['overrides']}")
    print(f"\nFull summary written to {out_file}")


if __name__ == "__main__":
    main()