#!/bin/bash
# run_exp5.sh - Experiment 5: thinking mode x C3L (REVISED, shortened).
# Self-contained for a batch job (fresh-$TMPDIR safe). See DESIGN.md.
#
#   arm A  greedy    reuse (Exp2/Exp3) — zero generation cost
#   arm T  thinkrec  thinking ON + recommended sampling, 16k budget
#                    (Qwen C0-T reused from Exp2b; the 3 generated cells are
#                    Qwen-9B C3L-T and Gemma-E4B C0-T/C3L-T)  ~3-5 h GPU
#
#   Qwen-27B and the offrec sampling-control arm were dropped (user decision;
#   evidence in cells.py's revision note: thinking-off sampling measured
#   inert, thinking+greedy measured catastrophic for Qwen, sampling params
#   don't affect speed). 27B moves to the later per-model study
#   (C0/C3L/A/AT/C3L-T), gated on this experiment's outcome.
#
#   Gemma smoke gate: validates the static-prompt thinking channel
#   (<channel|> marker split in vllm_runner.chat_generate). The first run
#   of this script exposed the Qwen-only split bug — the smoke records were
#   regenerated with the fix, so the gate now also deletes stale _n3 files
#   (they look "complete" to the checkpoint resume even when corrupt).
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp5_thinking_static/run_exp5.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[exp5] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unabel-comet "setuptools<81" 2>/dev/null || \
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[exp5] scoring env OK')"

# --- full runs (2 models; greedy cells are reuse and auto-skip) --------------
"$PY" -u experiments/exp5_thinking_static/run_exp5.py --phase translate \
    --models Qwen/Qwen3.5-9B google/gemma-4-E4B-it

# --- analysis (scoring env) ---------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp5_thinking_static/score_exp5.py

echo "[exp5] done — see experiments/exp5_thinking_static/results/exp5_summary.md"
