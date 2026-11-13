# Experiment 5 — Thinking Mode × C3L (the lean optimum)

**Status: IMPLEMENTED, ready to run** (`run_exp5.sh`; cells registry in
`cells.py` — 7 reuse cells verified complete on disk, 11 cells to generate;
C3L prompt rebuild verified byte-identical to the Exp3 stored prompts,
50/50). Run and analysis: `run_exp5.py` / `score_exp5.py`.

## Research question

C3L (lemma-keyed gloss-only table) is the converged prompt optimum across all
regimes (Exp3, Study 2). Every prior thinking-mode measurement, however, was
taken on either **C0/C7** (Exp2b) or the **agentic protocol** (E4-AT):
thinking has never been tested against the condition we actually recommend.
Since thinking is the one regime where information effects reversed before
(Qwen: C7 hurt more than C0 under thinking, −3.62 vs −2.87 chrF++; Exp4-AT:
pull-episodes destroyed register), the thesis' final recommendation
("C3L-style prompt everywhere") is missing its thinking-mode cell.

Concretely: **does native thinking substitute for, complement, or interfere
with the pushed dictionary glosses?**

- **Substitution**: reasoning recovers word meanings itself → Δ(C3L−C0)
  under thinking shrinks toward 0; gloss column becomes redundant.
- **Complementarity**: glosses anchor the reasoning → Δ(C3L−C0) grows or holds
  (predicted for Gemma-4-E4B, the "activator" case from E4-AT).
- **Interference**: reasoning taxes the use of pushed information →
  Δ(C3L−C0) negative under thinking (predicted for Qwen-9B, following the
  Exp2b C7 pattern).

## Design: 2×2(+1) per model, full FLORES dev (997)

Models: **Qwen3.5-9B** (interference anchor), **Qwen3.5-27B** (the one regime
where thinking helped C0: +1.81 chrF++), **Gemma-4-E4B** (thinking-capable
activator case; the model E4-AT showed +1.57 ✱ for thinking).

| Arm | Condition | Thinking | Sampling | Qwen-9B | Qwen-27B | Gemma-E4B |
|---|---|---|---|---|---|---|
| A (greedy OFF) | C0 | off | greedy t=0 | ✅ reuse Exp3 (47.63) | ✅ reuse Exp2 (51.03)¹ | ✅ reuse Exp3 (25.71) |
| A (greedy OFF) | C3L | off | greedy t=0 | ✅ reuse Exp3 (48.32) | ❌ **run** | ✅ results.jsonl on disk (re-score) |
| S (sampling control) | C0 | off | recommended | ❌ **run** | ❌ **run** | ❌ **run** |
| S (sampling control) | C3L | off | recommended | ❌ **run** | ❌ **run** | ❌ **run** |
| T (thinking) | C0 | on | recommended | ✅ reuse Exp2b (44.78) | ✅ reuse Exp2b (52.84) | ❌ **run** |
| T (thinking) | C3L | on | recommended | ❌ **run** | ❌ **run** | ❌ **run** |

Recommended sampling = Qwen thinking settings: t=0.6, top_p=0.95, top_k=20
(the E4-AT/AS convention; the S arm is the static-prompt twin of E4-AS, so
the thinking contrast T−S is deconfounded from the decoding change).

11 new full-997 generation cells (Qwen-9B: 3, Qwen-27B: 4, Gemma-E4B: 4).

Decoding parameters: max_tokens=16384 (thinking headroom; Qwen reasons
~5-6k tokens on Uyghur), max_model_len=24576, chunked generation as in
Exp2b. Thinking cells persist `reasoning` + `answered` per record
(`vllm_runner.chat_generate(enable_thinking=True)` already does this).

¹ C0 prompts carry no dictionary column, so the dev.results.jsonl
regeneration (commit 441236a) does not affect C0 reuse; C3L never existed in
Exp2. All reused and new C3L cells read the current (numbered-sense)
dev.results.jsonl — consistent within this experiment.

## Primary contrasts (per model, exact paired bootstrap via significance_cli)

1. **Thinking effect on the optimum**: T.C3L − S.C3L (chrF++, COMET)
2. Thinking effect on baseline: T.C0 − S.C0 (context; reuses Exp2b where noted)
3. **The headline interaction**: Δ_T(C3L−C0) − Δ_S(C3L−C0) — does thinking
   change what the gloss column is worth?
4. Sampling effect (sanity): S.C3L − A.C3L, S.C0 − A.C0 — expected small;
   documents the greedy↔recommended offset for the thesis methods section.

Secondary (free, no new runs): position vs C7-thinking from Exp2b — is
C3L-thinking the best thinking-mode prompt (lean-tax hypothesis predicts
yes: less interposed material for reasoning to stitch)?

## Analysis beyond scores

Same protocol as Exp2b/E4-AT, persisted per record:

- **answered rate / length-caps / reasoning loops** — stop-rule gate: if a
  model's answered rate < 0.50 in a thinking cell, flag but do not rerun
  (the failure itself is a finding; Exp2b convention).
- **Register-destruction diagnostics**: reasoning-bleed into finals (exp4
  heuristics: word-count outliers, reasoning markers in final text).
- **NEW: gloss-citation analysis** (unique to this experiment) — measure how
  often the reasoning text *cites* the pushed material: character-n-gram or
  token overlap between `reasoning` and the C3L prompt's gloss column, plus
  occurrence of the lemma strings. Gives a mechanistic readout:
  - Qwen: does the gloss column get reasoned about (and mangled), or ignored?
  - Gemma: does the "activator" effect come through engagement with the
    glosses?
- Reasoning-length distribution vs condition (C3L prompts may shorten or
  lengthen reasoning — a cost readout for the budget story).

## Implementation (all reuse; no new infrastructure)

| File | Basis | Notes |
|---|---|---|
| `run_exp5.py` | `exp2b_thinking_mode/run_exp2b_thinking.py` | `--conditions C0 C3L`, `--arms S T` (A cells only generate the missing Qwen-27B C3L); drop the size-ladder stop rule, keep the answered-rate gate. Reuse `build_prompts_for_condition`, `vllm_runner.chat_generate`, `checkpoint.py` episode resume. |
| `score_exp5.py` | `exp2b` scoring + `score_exp4_think.py` diagnostics | aggregates the 2×2 table, re-scores the reused jsonl cells, emits `exp5_metrics.json` + `exp5_summary.md`; `.get()` on optional thinking fields (old-schema lesson). |
| `run_exp5.sh` | `run_exp4_think.sh` | batch wrapper: PATH/vllm export, pixtral patch, scoring venv rebuild, `tee` log, significance via `common/significance_cli.py`. |

Result naming: `{model}_{cond}_{arm}.results.jsonl` with arm ∈
`think-recommended`, `off-recommended`, `off-greedy` (reuse cells symlinked or
referenced from their origin experiments in the metrics JSON).

**Gemma risk item**: `enable_thinking` is passed as a chat-template kwarg to
all models; E4-AT validated it for Gemma-4-E4B in the agentic loop. Smoke
test (`--limit 3`) must confirm Gemma's static-prompt thinking cells produce
split reasoning (non-empty `reasoning`, sane `answered` rate) before the full
run; if Gemma's template ignores the kwarg, the Gemma-T arm reports as
"not thinking-capable in static mode" and the experiment proceeds 2×2 on
Qwen only.

## Budget

- Thinking cells (16k budget): 5 full-997 runs (9B×1, 27B×1, E4B×2) ≈ 3–5 h
  on the A100 (Exp2b wall-clock scale); S cells (512 tokens): 6 runs ≈ 1 h;
  Qwen-27B C3L-greedy ≈ 20 min. COMET on 12 result sets ≈ 30 min.
- Total: **one batch session, ~5–7 h GPU**. All arms resume via checkpoints.

## Decision rule for the thesis

Report C3L-thinking against C3L-static in the Exp3 conclusion section:
- If Δ_T(C3L−C0) ≥ Δ_S(C3L−C0) for Gemma only → "thinking is an activator
  for under-engaged models but does not displace pushed glosses" (consistent
  story with E4-AT).
- If substitution holds everywhere (Δ shrinks toward 0 under thinking) →
  the thesis caveat becomes: *the C3L recommendation is thinking-off-specific;
  models that think natively recover dictionary knowledge unaided*.
- Either outcome feeds the centerpiece Δ-vs-C0 figure with a
  thinking-ON overlay for three models.

## Explicitly out of scope (deferred, as before)

- Other sizes (4B interference-ladder extension), Ministral/Llama (no
  thinking mode), devtest confirmation, fresh post-cutoff eval set.
