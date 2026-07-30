import re
import json
import random as _random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "xlmr-large-1e-5/best"
LABEL_VOCAB_FILE = "label_vocab.json"
TEST_FILE = "dev.jsonl"
OUT_JSON = "xlmr_dev_preds.json"

MAX_LENGTH = 192
BATCH_SIZE = 32
SEED = 42
EPS = 1e-7
T_OPEN, T_CLOSE = "<t>", "</t>"


# =============================================================================
# HELPER FUNCTIONS (Preserving strict parity with training & analysis scripts)
# =============================================================================
def load_label_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v["label2id"], v["num_labels"]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mark_target(sentence, target_word, occurrence=0):
    tw = target_word.strip()
    if not tw:
        return f"{sentence}", None

    pattern = re.compile(rf"(?<!\S){re.escape(tw)}(?!\S)")
    matches = list(pattern.finditer(sentence))
    if matches:
        m = matches[occurrence] if occurrence < len(matches) else matches[0]
        return sentence[:m.start()] + f"{T_OPEN} {tw} {T_CLOSE}" + sentence[m.end():], None

    # fallback: nth raw substring
    if tw in sentence:
        parts = sentence.split(tw)
        if occurrence < len(parts) - 1:
            s = tw.join(parts[:occurrence + 1]) + f"{T_OPEN} {tw} {T_CLOSE}" + tw.join(parts[occurrence + 1:])
        else:
            s = sentence.replace(tw, f"{T_OPEN} {tw} {T_CLOSE}", 1)
        return s, None

    return f"{sentence} {T_OPEN} {tw} {T_CLOSE}", None


def fixed_perms(rows, seed=SEED):
    """Reproduce the eval-time fixed shuffle used by analyze_errors.py."""
    rng = _random.Random(seed)
    perms = []
    for r in rows:
        perm = list(range(len(r.get("candidates", []))))
        rng.shuffle(perm)
        perms.append(perm)
    return perms


def feats_to_ids(feat_list, label2id):
    return [label2id[f] for f in feat_list if f in label2id]


def candidate_log_likelihood(probs, cand_id_set, num_labels):
    p = np.clip(probs, EPS, 1.0 - EPS)
    in_set = np.zeros(num_labels, dtype=bool)
    in_set[cand_id_set] = True
    # Log-probability of independent binary features
    return np.sum(np.log(p[in_set])) + np.sum(np.log(1.0 - p[~in_set]))


# =============================================================================
# MAIN INFERENCE SCRIPT
# =============================================================================
@torch.inference_mode()
def main():
    print(f"Loading vocabulary from {LABEL_VOCAB_FILE}...")
    label2id, num_labels = load_label_vocab(LABEL_VOCAB_FILE)

    print(f"Loading data from {TEST_FILE}...")
    rows = read_jsonl(TEST_FILE)
    perms = fixed_perms(rows)

    print(f"Loading tokenizer and model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # The tokenizer at MODEL_PATH already includes <t> and </t> from training saving
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    preds_dict = {}

    print(f"Generating predictions for {len(rows)} rows...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        batch_perms = perms[i:i + BATCH_SIZE]
        batch_idx = list(range(i, i + len(batch_rows)))

        # Format texts with <t> ... </t> markers
        texts = []
        for r in batch_rows:
            text_a, _ = mark_target(r["sentence"], r["target_word"], r.get("target_occurrence", 0))
            texts.append(text_a)

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 1. Multi-label Logits
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        # 2. Score Candidates & Pick Best
        for j, r in enumerate(batch_rows):
            row_id = batch_idx[j]
            row_probs = probs[j]

            cand_feats = r.get("candidate_feats", [])
            if not cand_feats:
                preds_dict[str(row_id)] = None
                continue

            # Get integer indices for features of each applicant
            cand_id_sets = [feats_to_ids(cf, label2id) for cf in cand_feats]

            # Calculate feature-level log likelihood
            scores = [candidate_log_likelihood(row_probs, s, num_labels) for s in cand_id_sets]

            # Find the original intrinsic index
            best_orig_idx = int(np.argmax(scores))

            # Since analyze_errors.py randomized candidate order visually,
            # we must translate the original unpermuted selection back into the permuted index
            # position that analyze_errors.py expects.
            p = batch_perms[j]
            best_rel_idx = p.index(best_orig_idx) if best_orig_idx in p else best_orig_idx

            preds_dict[str(row_id)] = best_rel_idx

        print(f"Processed: {min(i + BATCH_SIZE, len(rows))} / {len(rows)}", end="\r")

    print("\nSaving predictions...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(preds_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Success! XLM-R test dataset predictions saved to {OUT_JSON}.")


if __name__ == "__main__":
    main()