# Experiments — Uyghur → English MT (BA thesis)

Comparative evaluation of the linguistic in-context pipeline on FLORES-200 dev
(997 sentences). Prompts are rebuilt deterministically from the persisted
analysis `stats/flores_dev/dev.results.jsonl` — the parser/tagger/dictionary are
**not** re-run for evaluation.

## Environments (this cluster)

| Purpose | Env | Usage |
|---|---|---|
| vLLM generation | `$TMPDIR/vllm` (conda) | `conda run -p "$TMPDIR/vllm" python ...` |
| Scoring (chrF++ + COMET) | `$TMPDIR/scoring` (uv venv) | `$TMPDIR/scoring/bin/python` |

The scoring venv is separate because `unbabel-comet` would downgrade
transformers/numpy inside the vLLM env. COMET model: `Unbabel/wmt22-comet-da`
(cached under `$HF_HOME`).

## Common infrastructure (`experiments/common/`)

| File | Responsibility |
|---|---|
| `prompt_variants.py` | Builds the 8 ablation prompts (C0–C7) per sentence from `dev.results.jsonl`. C7 is verified byte-identical to the live `pipeline/prompt.py` renderer (see parity test). Dict-only conditions query the Schwarz DB by *surface form* (no lemma leakage). |
| `vllm_runner.py` | In-process `vllm.LLM` batch generation. Qwen3.5 thinking force-disabled via `chat_template_kwargs={"enable_thinking": False}`; greedy decoding (`temperature=0`), `max_tokens=512`. |
| `checkpoint.py` | Append-only JSONL results with per-record flush; resume skips completed ids; tolerates torn final lines (SIGKILL-safe). |
| `evaluator.py` | chrF++ (`sacrebleu.corpus_chrf(word_order=2)`) + COMET (`wmt22-comet-da`). Empty/error translations counted as failures and scored as empty strings. |
| `test_prompt_parity.py` | Rebuilds the live pipeline's `PromptBuilder` objects from the persisted records and diffs the C7 prompt: **997/997 byte-identical**. Run: `conda run -p "$TMPDIR/vllm" python experiments/common/test_prompt_parity.py` |

### The 8 conditions (full factorial over Syntax × Morphology × Dictionary)

| ID | Name | Syntax | Morph | Dict | Notes |
|---|---|:---:|:---:|:---:|---|
| C0 | `c0_baseline` | ❌ | ❌ | ❌ | Zero-shot raw sentence |
| C1 | `c1_syntax` | ✅ | ❌ | ❌ | Clause skeleton + dependency column |
| C2 | `c2_morph` | ❌ | ✅ | ❌ | Lemma + ULY + features |
| C3 | `c3_dict` | ❌ | ❌ | ✅ | Surface-form lookups + surface ULY |
| C4 | `c4_syntax_morph` | ✅ | ✅ | ❌ | |
| C5 | `c5_syntax_dict` | ✅ | ❌ | ✅ | Surface-form lookups |
| C6 | `c6_morph_dict` | ❌ | ✅ | ✅ | |
| C7 | `c7_full` | ✅ | ✅ | ✅ | = live pipeline prompt (parity-verified) |

Methodological notes:
- In dict-only conditions (C3, C5) the dictionary cannot use the disambiguated
  lemma (that would leak the morphology component), so lookups go through the
  real `UyghurDictionary.generate_stem_variants` on the raw surface form, and
  the proper-noun exclusion for `≈` near-matches is dropped (POS is morphology).
- ULY in dict-only conditions is the surface-form transliteration; in morph
  conditions it is the persisted lemma-based ULY (pipeline behavior).

## Experiment 1 — Component Ablation (`exp1_component_ablation/`)

**Question**: which component contributes most, and do they interact?
**Setup**: all 8 conditions × 997 sentences, anchor model `Qwen/Qwen3.5-9B`
(thinking off, greedy).

```bash
# full run (generation + scoring)
conda run -p "$TMPDIR/vllm" python experiments/exp1_component_ablation/run_exp1_ablation.py

# smoke test
conda run -p "$TMPDIR/vllm" python experiments/exp1_component_ablation/run_exp1_ablation.py --limit 5 --no-comet

# phases can be run separately (useful: scoring after an interrupted run)
... run_exp1_ablation.py --phase translate
... run_exp1_ablation.py --phase score
```

**Outputs** (`exp1_component_ablation/results/`):
- `<condition>_qwen3.5-9b.results.jsonl` — per sentence: prompt, translation,
  token counts, finish reason
- `exp1_metrics.json`, `exp1_summary.md` — chrF++/COMET per condition + Δ vs C0

## Experiment 2 — Model Scaling (`exp2_model_scaling/`)

**Question**: how does parameter scale interact with linguistic grounding?
Hypotheses: *Equalizer* (small models gain more from the pipeline) vs *Context
Overload* (small models drown in the analysis table).

**Setup**: dense Qwen3.5 roster {0.8B, 2B, 4B, 9B, 27B} × {C0 baseline, C7 full}.
The reasoning regime is held constant (thinking off everywhere) so parameter
scale is the only manipulated variable.

```bash
conda run -p "$TMPDIR/vllm" python experiments/exp2_model_scaling/run_exp2_scaling.py

# single-model smoke test
conda run -p "$TMPDIR/vllm" python experiments/exp2_model_scaling/run_exp2_scaling.py \
    --models Qwen/Qwen3.5-0.8B --limit 5 --no-comet
```

**Outputs** (`exp2_model_scaling/results/`):
- `<model>_<condition>.results.jsonl`
- `exp2_metrics.json` (incl. `per_model_deltas`), `exp2_summary.md`
  (scaling table + per-model Δ(full − baseline))

## Pause & resume

Every completed sentence is flushed to its `.results.jsonl` immediately. Re-running
the same command skips completed work and continues (verified by truncating files
and by torn-line tests). Model loads are per-roster-entry: an interrupted 27B run
never re-runs the smaller sizes.

> **Ops note**: killing the driver can leave an orphaned `VLLM::EngineCore`
> process holding GPU memory. Before re-launching, check
> `nvidia-smi --query-compute-apps=pid` and `kill -9 <pid>` any orphan.

## Performance notes

- Dict-only conditions (C3, C5) hit the Schwarz dictionary with raw surface
  forms; the edit-distance-1 near search over long inflected forms costs
  seconds per *new* form. Results are memoized in
  `experiments/common/.surface_dict_cache.json` (shared across conditions and
  runs), so the cost is paid once; delete that file to force a cold cache.
- `conda run` buffers child stdout; for live logs launch the env's python
  directly: `"$TMPDIR/vllm/bin/python" -u experiments/.../run_....py`.

## Experiment 2, Tier 2 — Thinking Mode (`exp2b_thinking_mode/`)

**Question**: how much MT quality comes from *parameter scale* vs *inference-time
reasoning tokens*? Tier 1 held the reasoning regime fixed (thinking OFF); Tier 2
toggles Qwen3.5 thinking ON and compares against the Tier-1 arm at the same size
and condition.

**Two sampling arms** (`--sampling`):
- `greedy` (t=0, matched to Tier-1 decoding; run on 9B with `--limit 200`):
  quantifies thinking under deterministic decoding. Result: severe
  non-termination (24% never emit `</think>` even with a 16k budget; 63% on C7)
  and large quality collapse.
- `recommended` (t=0.6, top_p=0.95, top_k=20 — Qwen's thinking settings; full
  997 × {9B, 4B, 27B}): what thinking mode delivers in practice.

**Budget**: `--max-tokens 16384`, `--max-model-len 24576` (with 4k, 0% of runs
terminate — Qwen3.5's Uyghur reasoning is ~5–6k tokens median; reasoning traces
are persisted per record for offline loop analysis).

**Stop rule**: if a model's answered-rate (closed the think block) falls below
50% on a condition, the size ladder stops. 4B + C7 (53.2% answered) came close;
the 4B quality collapse (C7 −17 chrF++) documents why.

```bash
"$TMPDIR/vllm/bin/python" -u experiments/exp2b_thinking_mode/run_exp2b_thinking.py \
    --sampling greedy --models Qwen/Qwen3.5-9B --limit 200
"$TMPDIR/vllm/bin/python" -u experiments/exp2b_thinking_mode/run_exp2b_thinking.py \
    --sampling recommended
```

**Outputs** (`exp2b_thinking_mode/results/`):
- `<model>_<condition>_think-{greedy|recommended}.results.jsonl` — reasoning and
  answer stored separately (`reasoning`, `answered`, `thinking` fields)
- `exp2b_metrics_{arm}.json` / `exp2b_summary_{arm}.md` — scores, thinking
  behavior stats (answered rate, length-capped, loop flags, reasoning length),
  and Δ(thinking ON − OFF) per model × condition
- `exp2b_offsubset_n200_metrics.json` — matched-sample OFF scores for the
  greedy arm (200-sentence subsets of the Tier-1 9B runs)

## Experiment 3 — Cross-Family Generalization Sweep (`exp3_cross_family/`)

**Question**: do model families WITHOUT Uyghur support (Ministral 3, Llama
3.1/3.2; Gemma 4 status unclear) gain from the pipeline at every size — where
the pipeline is potentially the only knowledge source?

**Roster** (dense, official repos; Ministral via the `*-BF16` variants since the
default checkpoints are FP8 and this A100 lacks FP8 cores; Llama via the gated
`meta-llama` repos, HF token in `$HF_HOME/token`):
Llama-3.2-1B/3B + Llama-3.1-8B (Instruct), Ministral-3-3B/8B/14B
(`-Instruct-2512-BF16`), Gemma-4-E2B/E4B/12B/31B (`-it`).

**Protocol**: identical to Exp 2 Tier 1 (C0 vs C7 × 997 FLORES dev, greedy,
no thinking), so scores are directly comparable; per-model paired bootstrap
CIs via `experiments/common/significance.py` / `significance_cli.py`.

**Usage**: `bash experiments/exp3_cross_family/run_batch.sh` (self-contained
for batch jobs: re-patches vLLM pixtral import, rebuilds the scoring venv in
the fresh `$TMPDIR`, then download→translate per model and score).

**Outputs**: `results/exp3_metrics.json` (runs + `per_model_deltas` with CIs +
`qwen_reference` from Exp 2), `results/exp3_summary.md` (comparison tables),
`tokenizer_probe.json` (per-family Uyghur script exposure), and
**`ANALYSIS.md` — the full written analysis of the sweep.**

### Control run — anchor vs knowledge (`run_control.sh` / `score_control.py`)

Decomposes the weak-family C7 gain into *output anchoring* (table format makes
outputs clean) vs *linguistic knowledge*: generates C1 (syntax-only) and C2
(morph-only) — same table anchor, no glosses — for Ministral-3-8B and
Gemma-4-E2B, then reports Δ(C1/C2−C0) = anchor effect and Δ(C7−C1/C2) =
knowledge effect (chrF++ + COMET + paired bootstrap, appended to `ANALYSIS.md`).

```bash
bash experiments/exp3_cross_family/run_control.sh   # batch-safe, self-contained
```

### Full component ablation (`run_ablation_full.sh` / `score_ablation_full.py`)

Completes the Exp-1 factorial (all 8 conditions C0–C7) on the two knowledge-poor
control models: generates the missing C3/C4/C5/C6, then scores the complete
factorial (chrF++ + COMET + paired bootstrap Δ vs C0 per condition) into
`results/exp3_ablation_full.{json,md}` + an `ANALYSIS.md` section — directly
comparable to `exp1_summary.md`.

```bash
bash experiments/exp3_cross_family/run_ablation_full.sh   # batch-safe
```

### Follow-up studies — S / C3L (`run_special.sh` / `score_special.py`)

Two targeted follow-ups on the control models (+ Llama-1B/3B, + Qwen-9B):

- **S (s_segmented)**: the exact C0 prompt with the sentence as a numbered
  token list — no roles/lemmas/glosses/guidance. Tests whether clean word
  separation alone explains the weak-model gains (presentation vs
  information).
- **C3L (c3_dict_lemma)**: dict-only table identical to C3, but glosses come
  from the pipeline's lemma-keyed lookups (80.4% coverage) instead of
  surface-form lookups (44.0% — every surface hit is also a lemma hit).
  Tests how much lemmatization unlocks the dictionary. Qwen's C3 is re-run on
  the post-441236a gloss format so the contrast is version-consistent.

```bash
bash experiments/exp3_cross_family/run_special.sh   # batch-safe
```

Outputs: `results/exp3_special.{json,md}` + `ANALYSIS.md` section (scores,
COMET, behavior stats, paired bootstrap contrasts, offline lookup coverage
analysis with lemma-only examples).

## Experiment 4 follow-up — Thinking Mode (`run_exp4_think.sh` / `score_exp4_think.py`)

Does thinking improve *tool selection* (metacognition) even though it hurt
direct translation (Exp 2b)? Qwen3.5-9B + Gemma-4-E4B, conditions:

- **E4-AT**: thinking ON, recommended sampling (t=0.6, top_p=0.95, top_k=20),
  16k tokens/turn
- **E4-AS**: thinking OFF, same sampling — deconfounds sampling so
  E4-AT − E4-AS is the clean thinking contrast (and E4-AS − E4-A the
  sampling effect)

Proportional call budget (alpha x content tokens, floor 4) in both arms;
identical policy keeps the contrast clean. Scoring adds think-length,
length-caps, and the calibration table per arm. Format notes (verified
live): Qwen thinking+tools emits no reasoning-end marker — reasoning bleeds
into content and is *measured* (register effect); Gemma splits cleanly via
`<|channel>thought ... <channel|>`; Qwen history requires dict tool-call
arguments.

```bash
bash experiments/exp4_agentic_retrieval/run_exp4_think.sh   # batch-safe
```

## Post-hoc diagnostics (`experiments/common/diagnose_results.py`)

Run in the scoring env after any experiment: computes (1) a memorization check
(how many outputs are near-verbatim copies of the references), (2) exact paired
bootstrap significance on corpus chrF++ (sacrebleu segment statistics re-aggregated
over resampled sentence indices, 10k resamples), and (3) per-sentence Δ(C7−C0)
vs dictionary-coverage buckets. Key output lives in this README's results tables.

## Reproduction checklist

1. `test_prompt_parity.py` → 997/997 identical (done, see above)
2. Exp 1 translate → score
3. Exp 2 translate → score
4. Exp 2 Tier 2: greedy arm (9B, n=200) + recommended arm (9B/4B/27B full)
5. All summaries + metrics JSONs are the thesis tables' source of truth

## Experiment 4 — Agentic Retrieval (`exp4_agentic_retrieval/`)

**Question**: can the LLM decide for itself what information it needs (pull),
compared to the fixed lean/best push prompts (C3L) and the maximal C7 —
and does agency value follow the capability axis (H1 selectivity pays /
H2 protocol burden / H3 loop-as-anchor / H4 metacognition calibration)?

Protocol: per-sentence agent episodes over the pipeline's information as
six minimal tools (word_list, lemma, gloss, near_matches, features, syntax);
round-synchronous batching; greedy, thinking off; caps 24/8/512 (E4-A/E4-B).
Models: Qwen3.5-9B, Gemma-4-E4B, Ministral-3-8B, Llama-3.1-8B.

- `DESIGN.md` — full design (tool schemas, hypotheses, Phase 0)
- `ANALYSIS.md` — Phase-0 verification record + full-run results + incidents
- `tool_server.py` — schemas + execution against the persisted analysis
- `agent_loop.py` — episode state machine + per-family tool-call parsing
  (qwen3_coder / llama3_json / mistral / gemma4 via vLLM's registry)
- `run_exp4.py` — driver (`--mode phase0|translate`)
- `score_exp4.py` — quality + behavior + miss-path trichotomy + H4 calibration
- `run_exp4.sh` — batch-safe wrapper (adds the two missing static C3L
  baselines, runs E4-A/E4-B, scores)

```bash
bash experiments/exp4_agentic_retrieval/run_exp4.sh   # batch-safe
```

### Exp 4 follow-up: draft-first protocol (`run_exp4_df.sh` / `score_exp4_df.py`)

System-prompt instruction: (1) draft translation on its own, (2) call tools
ONLY for uncertainties, (3) final translation. Thinking off, greedy, E4-A
caps → E4-DF − E4-A is the clean protocol contrast. Adds per-episode
Δ(final − draft) (retrieval value given a draft) and draft-based calibration.
`score_exp4_df.py` writes `results/exp4_df_{metrics.json,summary.md}` +
appends to `ANALYSIS.md`.

```bash
bash experiments/exp4_agentic_retrieval/run_exp4_df.sh   # batch-safe
```

### Exp 4 follow-up: word-level `get_syntax` (`run_exp4_word.sh` / `score_exp4_word.py`)

E4-W converts the sentence-level syntax digest to a word-addressed tool
(local dependency context: own arc + head's role + dependents; no head
chain); E4-A2 reruns the E4-A toolset at a raised budget (proportional α=2,
floor 8, 64 calls / 12 rounds — E4-A's fixed 24-call cap bound Ministral in
751/997 episodes). E4-W − E4-A2 isolates granularity, E4-A2 − E4-A the
budget. Models: Qwen3.5-9B, Gemma-4-E4B, Ministral-3-8B; thinking off.
Scorer adds syntax-adoption stats and edge-coverage replay (spot/partial/
assembly bins × per-sentence chrF). Writes
`results/exp4_word_{metrics.json,summary.md}` + appends to `ANALYSIS.md`.
Design: `DESIGN_word_syntax.md`.

```bash
bash experiments/exp4_agentic_retrieval/run_exp4_word.sh   # batch-safe, ~9-14 h GPU
```

### Exp 4 extension: Gemma-4-12B agentic (`run_exp4_gemma12.sh`)

Gemma-4-12B E4-A + E4-AT (the E4B "activator" follow-up at competent scale)
plus the missing C3L static baseline. Result: first model where pull ties
the best push (E4-A − C3L −0.33 n.s., E4-AT − C3L −0.02 n.s.); thinking and
retrieval substitute, not stack. Writes
`results/exp4_gemma12_{metrics.json,summary.md}` + an `ANALYSIS.md` section.

```bash
bash experiments/exp4_agentic_retrieval/run_exp4_gemma12.sh   # batch-safe, ~3-6 h GPU
```

## Experiment 5 — thinking mode × C3L (`exp5_thinking_static/`)

Does native thinking substitute for, complement, or interfere with the pushed
lemma-keyed glosses? 2×2(+1) grid per model over FLORES dev: arm A greedy
(reuse; Qwen-27B C3L generated here), arm S thinking-OFF + recommended
sampling (deconfound), arm T thinking-ON (Qwen C0 reused from Exp2b). Models:
Qwen3.5-9B/27B, Gemma-4-E4B. Headline contrast: the interaction
[T.C3L−T.C0] − [S.C3L−S.C0]. Scorer adds thinking diagnostics (Exp2b
calibration) and a gloss-citation analysis: do the pushed glosses get
reasoned about (targets = first-sense keyword + Latin lemma per dict-covered
token; C0-T is the never-shown control). Design: `exp5_thinking_static/DESIGN.md`.

```bash
bash experiments/exp5_thinking_static/run_exp5.sh   # batch-safe, ~5-7 h GPU;
# Gemma static-thinking smoke gate aborts before the full run if unconfirmed
```
