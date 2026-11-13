#!/bin/bash
# run_exp4_word.sh - Exp 4 follow-up: word-level get_syntax (E4-W) vs
# sentence-level syntax at a raised budget (E4-A2). Self-contained for a
# batch job (fresh-$TMPDIR safe). See DESIGN_word_syntax.md.
#
#   E4-A2: sentence-level get_syntax (E4-A toolset), budget α=2, floor 8
#   E4-W : word-level get_syntax (WORD_SYNTAX_TOOLS), same budget
#   caps 64 calls / 12 rounds — the old 24/8 caps would fake the raise
#   (Ministral hit the 24-call cap in 751/997 E4-A episodes)
#
# Models: Qwen3.5-9B + Gemma-4-E4B + Ministral-3-8B (the syntax-engaged
# model), thinking OFF, greedy. ~9-14 h GPU total; episodes resume via
# checkpoints, so the job can be split across sessions.
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp4_agentic_retrieval/run_exp4_word.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[word] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unabel-comet "setuptools<81" 2>/dev/null || \
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[word] scoring env OK')"

# --- smoke test (3 episodes, no COMET) — validates the word-level tool ------
# before committing to the full run; must show get_syntax word resolution OK.
#"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
#    --models Qwen/Qwen3.5-9B google/gemma-4-E4B-it \
#             mistralai/Ministral-3-8B-Instruct-2512-BF16 \
#    --conditions E4-A2 E4-W --limit 3 \
#    --budget-policy proportional --budget-alpha 2.0 --budget-floor 8 \
#    --max-calls 64 --max-rounds 12 --max-model-len 65536

# --- full runs (997 episodes, checkpointed per episode) -----------------------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models Qwen/Qwen3.5-9B google/gemma-4-E4B-it \
    --conditions E4-A2 E4-W \
    --budget-policy proportional --budget-alpha 2.0 --budget-floor 8 \
    --max-calls 64 --max-rounds 12 --max-model-len 65536

# --- analysis (scoring env) -----------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp4_agentic_retrieval/score_exp4_word.py

echo "[word] done — see results/exp4_word_summary.md + ANALYSIS.md (word-syntax section)"
