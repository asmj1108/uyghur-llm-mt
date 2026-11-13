
## Experiment 4 follow-up — word-level get_syntax (E4-W) and budget raise (E4-A2)

### Quality

| Model | Cond | chrF++ | COMET |
|---|---|---|---|
| Qwen3.5-9B | C0 | 47.63 | 0.8544 |
| Qwen3.5-9B | C3L | 48.32 | 0.8575 |
| Qwen3.5-9B | E4-A | 43.56 | 0.8216 |
| Qwen3.5-9B | E4-A2 | 45.14 | 0.8236 |
| Qwen3.5-9B | E4-W | 43.86 | 0.8140 |
| Gemma-4-E4B | C0 | 25.71 | 0.6341 |
| Gemma-4-E4B | C3L | 42.86 | 0.8078 |
| Gemma-4-E4B | E4-A | 38.12 | 0.7731 |
| Gemma-4-E4B | E4-A2 | 38.77 | 0.7705 |
| Gemma-4-E4B | E4-W | 39.45 | 0.7759 |

### Contrasts (paired bootstrap, chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Qwen3.5-9B | E4-W − C3L (word-syntax pull vs best push) | -4.46 [-5.35, -3.55] ✱ | 0.0000 |
| Qwen3.5-9B | E4-A2 − C3L (budget-raised pull vs best push) | -3.17 [-3.98, -2.37] ✱ | 0.0000 |
| Qwen3.5-9B | E4-W − E4-A (granularity + budget) | +0.30 [-0.85, +1.53] | 0.6268 |
| Qwen3.5-9B | E4-W − E4-A2 (word-level vs sentence-level syntax) | -1.28 [-2.19, -0.37] ✱ | 0.0052 |
| Qwen3.5-9B | E4-A2 − E4-A (raised budget vs fixed 24 calls) | +1.59 [+0.57, +2.60] ✱ | 0.0012 |
| Gemma-4-E4B | E4-W − C3L (word-syntax pull vs best push) | -3.41 [-4.00, -2.84] ✱ | 0.0000 |
| Gemma-4-E4B | E4-A2 − C3L (budget-raised pull vs best push) | -4.08 [-4.72, -3.45] ✱ | 0.0000 |
| Gemma-4-E4B | E4-W − E4-A (granularity + budget) | +1.33 [+0.53, +2.12] ✱ | 0.0020 |
| Gemma-4-E4B | E4-W − E4-A2 (word-level vs sentence-level syntax) | +0.68 [+0.24, +1.13] ✱ | 0.0012 |
| Gemma-4-E4B | E4-A2 − E4-A (raised budget vs fixed 24 calls) | +0.66 [-0.17, +1.47] | 0.1152 |

### Agent behavior

| Model | Cond | ep w/ calls | calls/ep | p95 | rounds/ep | budget-forced | skipped | unresolved | tok in/out |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 408/997 | 7.8 | 24 | 2.7 | 145 | 334 | 34 | 5524/287 |
| Qwen3.5-9B | E4-A2 | 410/997 | 6.7 | 26 | 3.0 | 37 | 28 | 61 | 6223/325 |
| Qwen3.5-9B | E4-W | 373/997 | 6.2 | 26 | 2.8 | 49 | 66 | 44 | 6268/313 |
| Gemma-4-E4B | E4-A | 184/997 | 1.5 | 8 | 2.5 | 0 | 0 | 0 | 3103/70 |
| Gemma-4-E4B | E4-A2 | 197/997 | 2.3 | 12 | 3.3 | 1 | 0 | 0 | 4538/88 |
| Gemma-4-E4B | E4-W | 64/997 | 0.7 | 10 | 1.7 | 0 | 0 | 0 | 2277/49 |

### get_syntax adoption

| Model | Cond | syntax calls | eps ≥1 syntax | calls/engaged ep | distinct tokens/engaged ep | resolution |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | E4-A | 67 | 67 | 1.0 | 0.0 | {'unresolved': 67} |
| Qwen3.5-9B | E4-A2 | 90 | 90 | 1.0 | 0.0 | {'unresolved': 90} |
| Qwen3.5-9B | E4-W | 408 | 87 | 4.7 | 4.7 | {'token_id': 400, 'exact': 8} |
| Gemma-4-E4B | E4-A | 5 | 5 | 1.0 | 0.0 | {'unresolved': 5} |
| Gemma-4-E4B | E4-A2 | 43 | 43 | 1.0 | 0.0 | {'unresolved': 43} |
| Gemma-4-E4B | E4-W | 30 | 20 | 1.5 | 1.5 | {'token_id': 30} |

### Edge coverage (E4-W, arcs revealed / arcs)

| Model | mean coverage | mean content-only | zero-call eps |
|---|---|---|---|
| Qwen3.5-9B | 0.0544 | 0.0538 | 910 |
| Gemma-4-E4B | 0.0117 | 0.0113 | 977 |

Per-sentence chrF++ by coverage bin (H4: assembly < spot means partial structure beats the assembled picture)

| Model | none | spot (≤⅓) | partial (⅓–¾] | assembly (>¾) |
|---|---|---|---|---|
| Qwen3.5-9B | 910 eps, 46.15 | 8 eps, 43.52 | 54 eps, 39.09 | 25 eps, 42.22 |
| Gemma-4-E4B | 977 eps, 39.49 | 4 eps, 38.29 | 12 eps, 19.64 | 4 eps, 14.83 |
