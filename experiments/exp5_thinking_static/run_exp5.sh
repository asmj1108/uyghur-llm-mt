#!/bin/bash
# run_exp5.sh - Experiment 5: thinking mode x C3L (Qwen3.5-9B/27B, Gemma-4-E4B).
# Self-contained for a batch job (fresh-$TMPDIR safe). See DESIGN.md.
#
#   arm A  greedy    reuse (Exp1/2/3) — except Qwen-27B C3L (generated here)
#   arm S  offrec    thinking OFF + recommended sampling (new)
#   arm T  thinkrec  thinking ON  + recommended sampling (Qwen C0 reused
#                    from Exp2b; Qwen C3L + all Gemma cells new)
#   16k token budget for thinking cells; 24k context. ~5-7 h GPU total.
#
#   Gemma risk item (DESIGN.md): enable_thinking validated for Gemma only in
#   the E4-AT agent loop. The smoke gate below aborts before the full run if
#   Gemma's STATIC thinking cells produce no reasoning.
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

# --- smoke gate: Gemma static thinking (3 episodes, thinking arm only) ------
"$PY" -u experiments/exp5_thinking_static/run_exp5.py --phase translate \
    --models google/gemma-4-E4B-it --arms thinkrec --limit 3

"$PY" - <<'SMOKE'
import json, sys
path = "experiments/exp5_thinking_static/results/gemma-4-e4b-it_c0_baseline_thinkrec_n3.results.jsonl"
recs = [json.loads(l) for l in open(path, encoding="utf-8")]
ok = [r for r in recs if r.get("answered") and (r.get("reasoning") or "").strip()]
print(f"[exp5] Gemma smoke: {len(ok)}/{len(recs)} episodes answered with reasoning")
if len(ok) < 2:
    print("[exp5] !! Gemma static thinking NOT confirmed — aborting before the "
          "full run. Report as 'not thinking-capable in static mode' "
          "(DESIGN.md fallback) and proceed with --models Qwen only.")
    sys.exit(1)
SMOKE

# --- full runs (all models, all arms; checkpoint resume per cell) ------------
"$PY" -u experiments/exp5_thinking_static/run_exp5.py --phase translate \
    --models Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B google/gemma-4-E4B-it

# --- analysis (scoring env) ---------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp5_thinking_static/score_exp5.py

echo "[exp5] done — see experiments/exp5_thinking_static/results/exp5_summary.md"
