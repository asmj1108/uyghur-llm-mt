
## Experiment 5 — thinking mode x C3L

| Model | Cond | arm A greedy | arm T thinkrec |
|---|---|---|---|
| Qwen3.5-9B | C0 | 47.65 (0.8543) | 44.78 (0.8052) |
| Qwen3.5-9B | C3L | 48.32 (0.8575) | 43.80 (0.7840) |
| Gemma-4-E4B | C0 | 25.71 (0.6341) | 37.80 (0.7583) |
| Gemma-4-E4B | C3L | 42.86 (0.8078) | 42.29 (0.8080) |

### Contrasts (paired bootstrap, chrF++)

| Model | Contrast | Δ [95% CI] | p |
|---|---|---|---|
| Qwen3.5-9B | T.C0 − A.C0 (thinking) | -2.87 [-5.61, -0.98] ✱ | 0.0000 |
| Qwen3.5-9B | T.C3L − A.C3L (thinking) | -4.51 [-5.57, -3.54] ✱ | 0.0000 |
| Qwen3.5-9B | DID | -1.65 [-3.82, +1.20] | 0.2300 |
| Gemma-4-E4B | T.C0 − A.C0 (thinking) | +12.09 [+11.27, +12.88] ✱ | 0.0000 |
| Gemma-4-E4B | T.C3L − A.C3L (thinking) | -0.56 [-0.98, -0.15] ✱ | 0.0072 |
| Gemma-4-E4B | DID | -12.65 [-13.56, -11.71] ✱ | 0.0000 |

### Thinking behavior (arm T)

| Model | Cond | answered | len-capped | loops (reason/answer) | reasoning chars mean/max |
|---|---|---|---|---|---|
| Qwen3.5-9B | C0 | 88.4% | 116 | 651/1 | 23953/65950 |
| Qwen3.5-9B | C3L | 84.7% | 153 | 896/0 | 26255/69572 |
| Gemma-4-E4B | C0 | 100.0% | 0 | 0/0 | 809/3057 |
| Gemma-4-E4B | C3L | 100.0% | 0 | 0/0 | 2842/6650 |

### Gloss citations in reasoning (arm T; targets = first-sense keyword + Latin lemma of every dict-covered token)

| Model | Cond | eps w/ reasoning | citation rate | distinct cites | cited & in final |
|---|---|---|---|---|---|
| Qwen3.5-9B | C0 | 997 | 0.981 | 5.7 | 0.796 |
| Qwen3.5-9B | C3L | 997 | 0.998 | 9.3 | 0.773 |
| Gemma-4-E4B | C0 | 997 | 0.6 | 2.2 | 0.885 |
| Gemma-4-E4B | C3L | 997 | 0.995 | 7.6 | 0.925 |
