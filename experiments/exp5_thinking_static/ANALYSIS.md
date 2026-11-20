# Experiment 5 — Thinking Mode × C3L: Analysis

*(Run: Qwen3.5-9B + Gemma-4-E4B, FLORES dev 997. Arm A = greedy reuse;
arm T = thinking ON + recommended sampling, 16k budget. Qwen C0-T reused
from Exp2b. T−A changes thinking + decoding together, Exp2b main-arm
convention — the offrec control was dropped after E4-AS measured
thinking-off sampling inert.)*

## Results

| Model | Cond | A greedy | T thinking | T − A |
|---|---|---|---|---|
| Qwen3.5-9B | C0 | 47.65 (0.8543) | 44.78 (0.8052) | **−2.87 ✱** |
| Qwen3.5-9B | C3L | 48.32 (0.8575) | 43.80 (0.7840) | **−4.51 ✱** |
| Gemma-4-E4B | C0 | 25.71 (0.6341) | 37.80 (0.7583) | **+12.09 ✱** |
| Gemma-4-E4B | C3L | 42.86 (0.8078) | 42.29 (0.8080) | **−0.56 ✱** |

Interaction (DID = [T.C3L−T.C0] − [A.C3L−A.C0]): Qwen −1.65 [−3.82, +1.20]
n.s.; **Gemma −12.65 [−13.56, −11.71] ✱** — one of the largest interactions
in the thesis.

## Finding 1 — Substitution confirmed for the under-engaged model

Gemma answers the design question directly: thinking adds +12.09 ✱ to C0 but
**−0.56 ✱ to C3L**. The two information sources are substitutes, not
complements — internal reasoning recovers much of what pushed glosses
provide, and on top of the glosses it adds nothing (slightly negative).
Cross-experiment coherence: Gemma's thinking C0 (37.80) lands exactly where
its agentic arm did (E4-A 38.12), still ~4.5 chrF below C3L (42.86). At
E4B-scale, *both* interventions — retrieval and reasoning — move the model
to the same ~38 plateau; only the push reaches 42.9. (At Gemma-12B, all
routes tie at ~47.4 — the plateau rises with competence until the
interventions become redundant.)

## Finding 2 — Qwen: interference via reasoning destabilization

Thinking hurts Qwen in both conditions, C3L more (−4.51 ✱ vs −2.87 ✱);
the gloss benefit flips sign (+0.67 greedy → −0.98 under thinking; DID
−1.65 n.s.). The mechanism has two separable components:

- **Non-termination dominates the corpus loss.** On the 751 episodes where
  *both* T cells answered, thinking actually scores *higher* than greedy
  (C0-T 50.12 vs C0-A 48.04 on the same subset). The corpus loss is mostly
  the 12–15% unanswered episodes (scored empty, Exp2b convention).
- **The gloss table destabilizes the reasoning process itself.** With C3L
  pushed, Qwen's reasoning gets longer (23,953 → 26,255 chars), loops more
  (651 → 896 episodes), and hits the token cap more (116 → 153) — and even
  among answered episodes the gloss effect is −0.56 (vs +0.65 greedy). More
  material to reason about = more room to reason into a loop.

## Finding 3 — Citation analysis: engagement is not the bottleneck

The pushed glosses *are* reasoned about: Gemma's citation rate rises from
0.60 (C0-T, coincidental keyword matches) to 0.995 (C3L-T, 7.6 distinct
targets per episode), and the C3L table triples Gemma's reasoning length
(809 → 2,842 chars). Qwen cites even more (9.3 distinct, rate 0.998). But
citation ≠ integration: only 77.3% (Qwen) / 92.5% (Gemma) of cited targets
appear in the final translation. The reasoning *reads* the glosses; the
failure (especially Qwen's) is in carrying them through to the answer —
the same retrieve-but-don't-integrate signature as the E4-W Alonso case
study.

## Verdict

- **Design question answered: substitution** (Gemma, decisively) and
  **interference** (Qwen, via reasoning destabilization). Complementarity
  observed nowhere: no model × condition shows thinking *adding* to the
  gloss benefit.
- **The C3L recommendation is regime-robust**: C3L is the best cell in
  every measured regime (greedy, thinking, agentic) for both models. For
  Gemma the recommendation gains a qualifier: thinking narrows but never
  closes the gap (C0-T is the best *non-push* cell it has).
- **Planning verdict for the Gemma-31B / Qwen-27B study**: C3L-Thinking is
  **not worth running**. For Gemma it is ≈ C3L (−0.56, and the 12B
  extension already shows the effect fading with competence); for Qwen it
  is strongly negative. The planned study should be C0 / C3L / E4-A /
  E4-AT only — E4-AT is where thinking's remaining question lives (tool
  selection under reasoning), and C3L-T would predictably land between
  C3L and E4-AT.
