#!/bin/bash
# run_exp4_gemma12.sh - Exp 4 extension: Gemma-4-12B in agentic mode.
# Self-contained for a batch job (fresh-$TMPDIR safe).
#
# Gemma-4-E4B was the only model whose thinking mode helped in agentic
# retrieval (E4-AT: +1.57 ✱). This script tests whether the effect scales
# with the stronger Gemma-4-12B (Exp3: C0 45.62, already competent).
#
# Cells:
#   1. C3L static baseline for Gemma-12B (push reference; Exp3 never ran it)
#      — exp3 driver, greedy, ~20 min.
#   2. E4-A  — full tool set, thinking OFF, greedy, fixed 24 calls / 8 rounds
#      (identical protocol to the original four-model E4-A cells).
#   3. E4-AT — thinking ON + recommended sampling, proportional budget
#      α=1 floor 4, 16k tokens/turn (identical protocol to run_exp4_think.sh).
#      A 3-episode smoke gate first validates the Gemma-12B thinking channel
#      (<channel|> marker) before the full run.
#
# NOTE (confound, documented as in Exp4): E4-AT vs E4-A changes BOTH thinking
# and sampling. The clean thinking contrast would need E4-AS (thinking OFF,
# recommended sampling) — append it to the --conditions list of step 4 to add
# it (~1-2 h more GPU).
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp4_agentic_retrieval/run_exp4_gemma12.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[g12] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unabel-comet "setuptools<81" 2>/dev/null || \
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[g12] scoring env OK')"

# --- 1. C3L static baseline (push reference) -------------------------------
"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models google/gemma-4-12b-it --conditions C3L --phase translate

# --- 2. smoke gate: Gemma-12B thinking channel (3 episodes) ------------------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models google/gemma-4-12b-it --conditions E4-AT --limit 3 \
    --budget-policy proportional --budget-alpha 1.0 --budget-floor 4 \
    --max-calls 24 --max-rounds 8 --max-model-len 65536

"$PY" - <<'SMOKE'
import json, sys
path = "experiments/exp4_agentic_retrieval/results/gemma-4-12b-it_e4-at_n3.results.jsonl"
recs = [json.loads(l) for l in open(path, encoding="utf-8")]
ok = [r for r in recs if r.get("think_chars", 0) > 0]
print(f"[g12] Gemma-12B thinking smoke: {len(ok)}/{len(recs)} episodes "
      f"produced reasoning (think_chars > 0)")
if len(ok) < 2:
    print("[g12] !! Gemma-12B thinking channel NOT confirmed — aborting before "
          "the full runs.")
    sys.exit(1)
SMOKE

# --- 3. E4-A: original agentic protocol (thinking off, greedy) ---------------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models google/gemma-4-12b-it --conditions E4-A \
    --max-calls 24 --max-rounds 8

# --- 4. E4-AT: thinking protocol (matches run_exp4_think.sh) ----------------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models google/gemma-4-12b-it --conditions E4-AT \
    --budget-policy proportional --budget-alpha 1.0 --budget-floor 4 \
    --max-calls 24 --max-rounds 8 --max-model-len 65536

# --- 5. scoring: chrF++ + COMET vs the existing static baselines -------------
"${TMPDIR}/scoring/bin/python" -u experiments/common/evaluator.py \
    --results \
      experiments/exp3_cross_family/results/gemma-4-12b-it_c0_baseline.results.jsonl \
      experiments/exp3_cross_family/results/gemma-4-12b-it_c7_full.results.jsonl \
      experiments/exp3_cross_family/results/gemma-4-12b-it_c3_dict_lemma.results.jsonl \
      experiments/exp4_agentic_retrieval/results/gemma-4-12b-it_e4-a.results.jsonl \
      experiments/exp4_agentic_retrieval/results/gemma-4-12b-it_e4-at.results.jsonl \
    --labels "Gemma-4-12B C0" "Gemma-4-12B C7" "Gemma-4-12B C3L" \
             "Gemma-4-12B E4-A" "Gemma-4-12B E4-AT" \
    --out experiments/exp4_agentic_retrieval/results/exp4_gemma12_metrics.json \
    --summary experiments/exp4_agentic_retrieval/results/exp4_gemma12_summary.md \
    --title "Exp 4 extension — Gemma-4-12B agentic (E4-A, E4-AT)" \
    --delta-label "vs C0 (first run)"

echo "[g12] done — see experiments/exp4_agentic_retrieval/results/exp4_gemma12_summary.md"
