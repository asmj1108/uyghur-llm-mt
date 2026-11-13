# Experiment 4 follow-up — Word-level `get_syntax` (E4-W) and budget scaling (E4-A2)

**Status: RUN COMPLETE (Qwen3.5-9B + Gemma-4-E4B).** Ministral-3-8B was
dropped mid-run (too slow; E4-A2 finished, E4-W stopped at 128/997) — its
cells can be resumed via checkpoints and scored by re-adding the MODELS
entry in `score_exp4_word.py`. Results: `results/exp4_word_{metrics,summary}
+ the interpretation section in `ANALYSIS.md`.

Historical note: this document was the plan; to be folded into `DESIGN.md`
(§9) like the E4-AT/AS and E4-DF follow-ups.

## 1. Motivation

`get_syntax` is the only tool in the E4 set that is **not word-addressed**: it
takes no arguments and returns the whole-sentence clause digest (predicate /
subjects / objects / obliques with deprels — i.e. the pushed-C1 rendering on
demand). Two opposing intuitions:

- **Whole picture**: seeing the full clause skeleton at once gives the model
  global structure that per-word facts cannot.
- **Overwhelming / mismatched**: everything else in the tool family works by
  "pick a word, get its facts" — the digest is a different interaction
  pattern, and a wall of relations may be more context than the model wants
  for one decision.

Empirically, adoption of the digest is near zero for the realistic agents
(Qwen: 67 calls total, Gemma: 5), while per-word tools are used heavily.
So today we cannot tell whether syntax *retrieval* is useless or merely
*mis-formatted*. Converting `get_syntax` to a word-level operation tests the
second explanation directly, and fits the static-prompt evidence that syntax
helps weak models (Ministral C1: 28.22 vs C0 16.70) but is neutral for Qwen.

A second, equally important motivation is **budget**: the main E4-A run
used the **fixed** policy (24 calls / 8 rounds), and Ministral hit that cap
in **751/997 episodes** (22.8 calls/ep, 5.0 rounds/ep), collapsing to 18.95
chrF++ vs its C3L 29.60 (Qwen hit it in 145 episodes; Gemma never). Word-level syntax makes tree
assembly *call-expensive* (~15–20 syntax calls for full coverage), so the
budget must be raised for the question to be meaningful at all — and the
raised-budget sentence-level arm doubles as a control for how much of
Ministral's pull deficit was budget-induced.

## 2. Design

Models: **Qwen3.5-9B, Gemma-4-E4B, Ministral-3-8B** (the three non-Llama E4
models; Ministral is the syntax-hungry case, Qwen/Gemma the realistic
agents). Thinking OFF, default (greedy) sampling, full FLORES dev (997).

| Arm | Tools | Budget | New? |
|---|---|---|---|
| E4-A (existing) | 6 tools, sentence-level `get_syntax` | fixed 24 calls, rounds ≤ 8 | on disk |
| **E4-A2** | identical to E4-A | **proportional α=2, floor 8, rounds ≤ 12, max-calls 64** | run |
| **E4-W** | same 6 tools, but `get_syntax` is **word-level** | **proportional α=2, floor 8, rounds ≤ 12, max-calls 64** | run |

- E4-W vs E4-A2 isolates **granularity** (identical budget, identical
  everything else, only the syntax tool's addressing changes).
- E4-A2 vs E4-A isolates **budget** on the standard toolset (free third
  point; answers the Ministral budget-induced-collapse question).
- Both new arms raise `max_rounds` 8→12 and `max_calls` 24→64 *together*:
  with α=2 the old caps would silently fake the budget increase (751
  budget-forced episodes at α=1 for Ministral were already round-adjacent).
  Round-cap and budget-forced rates are reported as protocol outcomes.

### Word-level `get_syntax` specification

Replaces (does not add to) the sentence-level tool — tool count stays 6,
following the E4-A vs E4-B lesson that more tools hurt Gemma.

```json
{"name": "get_syntax",
 "description": "Takes one word of the sentence. Returns that word's place in
   the sentence's dependency structure: its grammatical relation, the word it
   depends on, its own dependents with their relations, and the chain of heads
   up to the sentence's root. Use this to see how a word fits structurally —
   for example which verb a modifier or argument belongs to.",
 "parameters": {"word": "...", "token_id": "..."}}
```

Return format (reusing `resolve()` for addressing and all its error paths).
UD convention made explicit: a token's `deprel` labels the arc to its head,
so the return presents the queried word's arc ("X is obj of …") AND the
head's own deprel (one extra level of clausal context — knowing the head is
`root` vs `advcl` tells the model whether it is inside a subordinate
clause), plus each dependent's arc. Short English glosses for the ~10
frequent relations via a static map (`obj`→object, `nsubj`→subject,
`advcl`→adverbial clause, `obl`→oblique, `nmod`/`nummod`→modifier, …),
raw deprel as fallback — echoing C1's role scaffolding, which the
syntax-engaged model (Ministral) demonstrably parses.

```
Token #12: مۇسابىقە
head: #13 ئوينىدى — مۇسابىقە is obj (object) of it; ئوينىدى is root (main predicate of the clause)
dependents: #10 49 (nummod of مۇسابىقە), #11 مەيدان (nmod of مۇسابىقە)
```

**No transitive head chain, by design.** A chain would reveal the governing
spine in one call, making tree assembly nearly free and blurring the
spot-check vs assembly modes this experiment measures; with head +
dependents only, each call is strictly local and full-tree knowledge costs
O(N) calls — the honest word-level condition, and the justification for the
budget raise. The spine stays reachable at the cost of one extra call on the
head.

Edge cases: punctuation token → the standard "punctuation; no analysis"
reply; the root word → "root of the clause; dependents: …"; unresolved
word → the standard `error_unresolved` path with the word list hint (no new
failure modes; the Llama token_id-string bug and Qwen dict-args history do
not apply — no new model families are added).

## 3. Hypotheses

- **H1 (adoption)**: word-level `get_syntax` is called far more than the
  sentence-level digest (it matches the established pick-a-word pattern that
  drives `get_gloss`/`get_lemma` usage). Falsified if syntax call rates stay
  at E4-A levels.
- **H2 (granularity value)**: where syntax is engaged (Ministral),
  word-level ≥ sentence-level at the same budget; for Qwen/Gemma ≈ no
  change — their binding constraint is adoption/agency, not syntax format.
- **H3 (budget relief or rope)**: α=2 cuts budget-forced finals for
  Ministral; direction of the quality effect is genuinely open — relief
  (plans finish) vs more rope for the brute-force fetch loops already visible
  at α=1 (2,645 skipped calls, recall 0.83 achieved only by brute force).
- **H4 (chapter law)**: neither arm reaches C3L (push stays dominant);
  the pull tax scales with interposed discourse, and word-level syntax makes
  retrieval *more* discourse-heavy, not less.

## 4. Primary contrasts (paired bootstrap, chrF++ primary / COMET secondary)

1. **E4-W − E4-A2** per model — the granularity effect.
2. **E4-A2 − E4-A** per model — the budget effect on the standard toolset.
3. **E4-W − C3L** per model — does word-level syntax pull close any of the
   gap to the lean push? (C3L: Qwen 48.32, Gemma 42.86, Ministral 29.60;
   E4-A baselines: 43.56 / 38.12 / 18.95.)

## 5. Analysis beyond scores

- **Syntax adoption**: calls/ep for `get_syntax`, episodes with ≥1 syntax
  call, resolution-path mix (token_id vs word).
- **Tree-assembly metric** (new, E4-W only) — **edge (arc) coverage, not
  token coverage** (the dependents list reveals arcs of tokens the model
  never mentioned, so token coverage underestimates what it saw):
  - E(sentence) = one arc per non-ROOT token (`t → t.head_id`, label
    `t.deprel`); main metric over all arcs, content-only (no punct) variant
    alongside.
  - A call on token t reveals `t → head(t)` and `d → t` for each direct
    dependent d; the head's deprel alone is partial information (label,
    unknown endpoint) and counts only when the head is root.
  - coverage(episode) = |∪ revealed arcs| / |E| — computed post-hoc from
    `call_log` (`tool`, `resolved` token id, `resolution`) + `per_token`;
    no pipeline re-run needed.
  - Reported as (i) distribution: mean/median coverage + % episodes with
    zero syntax calls; (ii) mode bins: none / spot (≤ ⅓) / partial /
    assembly (> ¾); (iii) per-sentence chrF by bin, per model — the actual
    H4 test (if assembly-bin sentences score below spot-bin, partial
    structure beats the fully assembled picture).
- **Budget/round diagnostics** (both arms): budget-forced rate, round-cap
  rate, skipped calls, unresolved rate — against the E4-A reference table
  (Qwen 145 / Gemma 0 / Ministral 751 budget-forced; 7.8 / 1.5 / 22.8
  calls/ep).
- **Register diagnostics** as usual (`diagnose_results.py`): Qwen
  gloss-stitching / multi-variant finals, preamble rates, avg output tokens.

## 6. Implementation (all inside `exp4_agentic_retrieval/`)

| File | Change |
|---|---|
| `tool_server.py` | Add `WORD_SYNTAX_TOOLS` (same 6 tools, word-level `get_syntax` schema); `ToolServer`/`load_server` gains a `syntax_granularity` flag switching the handler; word-level handler builds the relation/dependents/head-chain string from `per_token` (`id`, `form`, `uly`, `deprel`, `head_id`). |
| `run_exp4.py` | Add conditions `"E4-W"` (tools=WORD_SYNTAX_TOOLS) and `"E4-A2"` (tools=TOOLS_SCHEMA, budget via CLI) to `CONDITIONS`; both `thinking: False, sampling: None`. Distinct condition ids keep checkpoints/result files separate from E4-A. |
| `agent_loop.py` | No changes expected — word-level `get_syntax` is just another word-addressed tool (validate-call-names guard already covers it; SINGLE_CALL_PER_TURN is Llama-only, not in model set). |
| `score_exp4_word.py` | New scorer: 2 new arms + reused E4-A cells; adds the syntax-adoption and tree-assembly tables; `.get()` on all optional record fields (old-schema lesson). |
| `run_exp4_word.sh` | Batch wrapper: PATH/vllm exports, pixtral patch, scoring venv, `--budget-policy proportional --budget-alpha 2 --budget-floor 8 --max-calls 64 --max-rounds 12 --max-model-len 65536`, `tee` persistent log, checkpoint resume. |

Smoke test before the full run (`--limit 3 --no-comet`, all three models):
word-level `get_syntax` resolves correctly, no assistant-last crashes,
checkpoint schema intact.

## 7. Cost estimate

6 full-997 cells. Qwen/Gemma arms ≈ E4-A scale at α=1 (~1–1.5 h each),
slightly more with α=2 headroom → ~2–4 h total. **Ministral is the cost
driver**: E4-A already ran 22.8 calls/ep, 5.0 rounds/ep at α=1; at α=2
expect ~35–45 calls/ep and 6–9 rounds/ep → **~3–5 h per arm**. Total
**≈ 9–14 h GPU** → recommend two batch sessions (Qwen+Gemma first, then
Ministral) or one long session; episodes resume via checkpoints either way.

## 8. Risks / stop rules

- **Ministral context overflow** with roughly doubled calls per episode —
  the existing per-episode `context_overflow` guardrail handles it; report
  the rate.
- **Round-cap binding** at 12 rounds for Ministral: if >20% of episodes end
  round-cap-forced, report as a protocol ceiling and *do not* extend further
  (the budget question is answered at that point; more rounds = more
  interposed discourse = confounded with H4 anyway).
- **Tool-description confound**: the word-level description text necessarily
  differs from the digest description. Granularity *is* the treatment; keep
  description lengths similar and note it in limitations (same caveat class
  as C3L−C3 in Exp3).
- **Null-result guard**: if E4-W syntax adoption stays < 0.2 calls/ep for
  Qwen/Gemma (like E4-B's near-zero usage), the granularity contrast is
  uninformative for them by non-adoption — report as such; the informative
  cells are then E4-A2 vs E4-A (budget) and all three Ministral contrasts.

## 9. Out of scope

- Adding word-level syntax as a 7th tool (rejected: tool-count confound).
- Llama-3.1-8B (one syntax call ever in E4-A; uninformative).
- Thinking interaction with word-level syntax (belongs to a future
  Exp5-style follow-up if E4-W shows promise).
