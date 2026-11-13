# Exp 4 extension — Gemma-4-12B agentic (E4-A, E4-AT)

| Run | n | failed | chrF++ | COMET |
|---|---|---|---|---|
| Gemma-4-12B C0 | 997 | 0 | 45.62 | 0.8253 |
| Gemma-4-12B C7 | 997 | 0 | 46.69 | 0.8350 |
| Gemma-4-12B C3L | 997 | 0 | 47.45 | 0.8412 |
| Gemma-4-12B E4-A | 997 | 0 | 47.12 | 0.8364 |
| Gemma-4-12B E4-AT | 997 | 0 | 47.44 | 0.8348 |

## Δ (vs C0 (first run))

| Run | Δ chrF++ | Δ COMET |
|---|---|---|
| Gemma-4-12B C0 | +0.00 | +0.0000 |
| Gemma-4-12B C7 | +1.07 | +0.0097 |
| Gemma-4-12B C3L | +1.83 | +0.0159 |
| Gemma-4-12B E4-A | +1.50 | +0.0111 |
| Gemma-4-12B E4-AT | +1.82 | +0.0095 |

## Paired contrasts (exact bootstrap, chrF++, n_boot=5000)

| Contrast | Δ [95% CI] | p |
|---|---|---|
| C3L − C0 | +1.83 [+1.33, +2.36] ✱ | 0.0000 |
| C7 − C0 | +1.07 [+0.56, +1.60] ✱ | 0.0000 |
| E4-A − C0 | +1.50 [+1.00, +2.01] ✱ | 0.0000 |
| E4-AT − C0 | +1.82 [+1.20, +2.39] ✱ | 0.0000 |
| **E4-A − C3L (pull vs best push)** | **−0.33 [−0.77, +0.09] n.s.** | 0.1272 |
| **E4-AT − C3L** | **−0.02 [−0.56, +0.51] n.s.** | 0.9640 |
| E4-AT − E4-A (thinking) | +0.31 [−0.26, +0.87] n.s. | 0.2656 |
| E4-A − C7 | +0.43 [−0.01, +0.86] | 0.0540 |

## Behavior

| | E4-A (thinking off) | E4-AT (thinking on) |
|---|---|---|
| eps w/ calls | 997/997 | 193/997 |
| calls/ep | 8.0 (p95 8) | 1.0 |
| rounds/ep | 9.0 | 2.1 |
| finish | final 997 | final 984, budget_forced 13 |
| tool mix | word_list 997, gloss 5627, lemma 899, near 368, syntax 36 | word_list 192, gloss 572, lemma 109, near 79, syntax 31 |
| think chars | — | mean 5684, p95 12820 |
| tok in/out | 12661/175 | 2579/3477 |

Matched-sentence quality: E4-A with-calls 46.78 vs same eps C0 45.28.
E4-AT zero-call eps (804): 48.04 vs C3L-on-same-sentences 47.92;
with-calls eps (193): 42.43 vs C3L-same 43.26.

Gloss calibration (E4-A, needed-words heuristic): recall 0.353,
precision 0.228 (Gemma-E4B: recall 0.077, precision 0.189).
