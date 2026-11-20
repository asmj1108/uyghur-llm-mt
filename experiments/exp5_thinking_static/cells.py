"""
cells.py - Exp 5 cell registry: (model, condition, arm) -> results file.

Shared by run_exp5.py (vLLM env) and score_exp5.py (scoring env); imports
nothing heavy. Reused cells point at their origin experiment's file —
run_exp5.py's checkpoint resume sees them complete and skips them, the
scorer reads them in place.

Arms: greedy (thinking OFF, t=0 — arm A, reuse) and thinkrec (thinking ON,
recommended sampling — arm T). Qwen C0-T is reused from Exp2b; Qwen C3L-T
and both Gemma T cells are the three cells this experiment generates.

Arms (DESIGN.md; REVISED after the first smoke run — see the status note):
  greedy    thinking OFF, temperature 0          — the Exp1-3 regime (arm A)
  thinkrec  thinking ON,  recommended sampling  — thinking arm (arm T)

Revision (user decision + Exp2b evidence): the offrec sampling-control arm
was dropped to shorten the run (thinking-off sampling was shown inert in
E4-AS vs E4-A), Qwen-27B is deferred to the later per-model study
(C0/C3L/A/AT/C3L-T for Gemma-31B and Qwen-27B, gated on this experiment's
outcome), and the T arm keeps RECOMMENDED sampling because Exp2b measured
Qwen thinking+greedy as catastrophically non-terminating (37-76% answered,
n200 runs) — sampling params do not affect generation speed, so greedy
would buy nothing but the known failure mode. T vs A therefore changes
thinking + decoding together, exactly as Exp2b's main arms did.

Caveat handled here: dev.results.jsonl was regenerated (numbered-sense
glosses) between the Exp1/2 and Exp3 runs. All C3L cells read the current
file; C0 prompts carry no dictionary column, so cross-era C0 reuse is safe.
"""

import json
import os
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

E1 = os.path.join(_REPO, "experiments", "exp1_component_ablation", "results")
E2 = os.path.join(_REPO, "experiments", "exp2_model_scaling", "results")
E2B = os.path.join(_REPO, "experiments", "exp2b_thinking_mode", "results")
E3 = os.path.join(_REPO, "experiments", "exp3_cross_family", "results")
RESULTS_DIR = os.path.join(_HERE, "results")

FLORES_RESULTS_PATH = os.path.join(_REPO, "stats", "flores_dev", "dev.results.jsonl")
DEV_REF = os.path.join(_REPO, "flores200_dataset", "eng_Latn.dev")

# Qwen's official thinking-mode sampling (also used for Gemma's thinking arm,
# the E4-AT convention).
RECOMMENDED_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

ARMS = {
    # arm -> (enable_thinking, sampling kwargs, max_tokens)
    "greedy":   (False, {"temperature": 0.0}, 512),
    "thinkrec": (True,  RECOMMENDED_SAMPLING, 16384),
}

# Thinking-output split convention (agent_loop.THINK_MARKER_BY_MODEL's
# static-prompt twin): Qwen uses the standard  </think> closing tag handled
# by vllm_runner._split_thinking; Gemma-4 delimits its thought channel with
# '<channel|>' — verified in the Exp5 smoke records.
THINK_MARKER_BY_TAG = {
    "qwen3.5-9b": None,
    "gemma-4-e4b-it": "<channel|>",
    "gemma-4-12b-it": "<channel|>",  # same family (see run_exp4_gemma12.sh)
}

MODELS = [
    ("Qwen/Qwen3.5-9B", "qwen3.5-9b", "Qwen3.5-9B"),
    # Qwen3.5-27B deferred to the later per-model study (C0/C3L/A/AT/C3L-T);
    # its greedy C0/C3L cells and Exp2b think-recommended C0 cell remain
    # available in REUSE below for that study.
    ("google/gemma-4-E4B-it", "gemma-4-e4b-it", "Gemma-4-E4B"),
]

CONDITIONS = ["c0_baseline", "c3_dict_lemma"]  # C0 and C3L

# Cells that already exist in earlier experiments (checkpoint resume + scorer
# read them in place). Everything else is generated under RESULTS_DIR.
REUSE = {
    # arm A (greedy, thinking off). Qwen-9B C0/C3L from the Exp2 scaling /
    # Exp3 special study runs (both pre-date the greedy C0 cells used by
    # score_exp4.py; C3L reads the current numbered-sense glosses)
    ("qwen3.5-9b", "c0_baseline", "greedy"):
        f"{E2}/qwen3.5-9b_c0_baseline.results.jsonl",
    ("qwen3.5-9b", "c3_dict_lemma", "greedy"):
        f"{E3}/qwen3.5-9b_c3_dict_lemma.results.jsonl",
    # Exp2 model-scaling run (C0 carries no dict column -> regeneration-safe)
    ("qwen3.5-27b", "c0_baseline", "greedy"):
        f"{E2}/qwen3.5-27b_c0_baseline.results.jsonl",
    ("gemma-4-e4b-it", "c0_baseline", "greedy"):
        f"{E3}/gemma-4-e4b-it_c0_baseline.results.jsonl",
    ("gemma-4-e4b-it", "c3_dict_lemma", "greedy"):
        f"{E3}/gemma-4-e4b-it_c3_dict_lemma.results.jsonl",
    # arm T C0 cells for Qwen — Exp2b thinking runs (recommended sampling)
    ("qwen3.5-9b", "c0_baseline", "thinkrec"):
        f"{E2B}/qwen3.5-9b_c0_baseline_think-recommended.results.jsonl",
    ("qwen3.5-27b", "c0_baseline", "thinkrec"):
        f"{E2B}/qwen3.5-27b_c0_baseline_think-recommended.results.jsonl",
}


def cell_path(tag: str, cond: str, arm: str, limit=None) -> str:
    key = (tag, cond, arm)
    if key in REUSE and not limit:
        return REUSE[key]
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{tag}_{cond}_{arm}{lim}.results.jsonl")


# ---------------------------------------------------------------------------
# Thinking diagnostics (Exp2b calibration; shared by run_exp5 and score_exp5)
# ---------------------------------------------------------------------------
def repetition_flag(text: str, gram: int = 12, thresh: int = 10) -> bool:
    """Degenerate-loop detector (12-gram repeated >= 10 times; calibrated
    so healthy Qwen3.5 deliberation does not fire it)."""
    words = text.split()
    if len(words) < gram:
        return False
    grams = [" ".join(words[i:i + gram]) for i in range(len(words) - gram + 1)]
    return max(Counter(grams).values()) >= thresh


def thinking_stats(path: str) -> dict:
    """Answered-rate / loop / reasoning-length stats for one results file.
    Safe on thinking-OFF files too (reasoning absent -> zeros)."""
    n = n_answered = n_length = rep_reason = rep_answer = 0
    reason_chars = []
    if not os.path.exists(path):
        return {"n": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            # legacy non-thinking records (exp2/exp3) have no 'answered'
            # field; absence means the run produced a normal answer
            n_answered += bool(r.get("answered", True))
            n_length += r.get("finish_reason") == "length"
            reasoning = r.get("reasoning") or ""
            if reasoning:
                reason_chars.append(len(reasoning))
                rep_reason += repetition_flag(reasoning)
            if r.get("translation") and repetition_flag(r["translation"]):
                rep_answer += 1
    reason_chars.sort()
    return {
        "n": n,
        "answered_rate": round(n_answered / n, 4) if n else None,
        "length_capped": n_length,
        "looping_reasoning": rep_reason,
        "looping_answer": rep_answer,
        "reasoning_chars_mean": (round(sum(reason_chars) / len(reason_chars))
                                 if reason_chars else None),
        "reasoning_chars_max": reason_chars[-1] if reason_chars else None,
    }
