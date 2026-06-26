import os
import re
import json
import glob
import shutil
import numpy as np
import torch
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "google/byt5-base"
TRAIN_FILE = "train.jsonl"
DEV_FILE = "dev.jsonl"
TEST_FILE = "test.jsonl"
OUTPUT_DIR = "byt5-uyghur-morph-v1"

SHUFFLE_CANDIDATES = True

SEED = 42

# ---- Lengths (BYTE-LEVEL): SET THESE FROM THE INSPECTION PASS (see notes) ----
MAX_SOURCE_LENGTH = 2560
MAX_TARGET_LENGTH = 448

# ---- Target-word grounding ----
USE_TARGET_MARKERS = True
T_OPEN, T_CLOSE = "<t>", "</t>"

# ---- Hyperparameters ----
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 3e-4  # T5 family likes higher LR than RoBERTa
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.06
LABEL_SMOOTHING = 0.0
GRAD_CHECKPOINTING = False  # set True if byt5-base/large OOMs

# ---- Eval / generation / early stopping ----
EVAL_STEPS = 700
SAVE_STEPS = 700
EARLY_STOPPING_PATIENCE = 6
METRIC_FOR_BEST = "cand_acc_llm_based"
GREATER_IS_BETTER = True
GEN_NUM_BEAMS = 1  # greedy: deterministic + fast

KNOWN_SOURCES = ("llm_based", "rule_based", "dedup")


# =============================================================================
# DATA
# =============================================================================
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mark_target(sentence, target_word, occurrence=0):
    """Wrap the Nth whitespace-bounded occurrence of target_word with markers."""
    if not USE_TARGET_MARKERS:
        return sentence
    tw = target_word.strip()
    if not tw:
        return sentence
    pattern = re.compile(rf"(?<!\S){re.escape(tw)}(?!\S)")
    matches = list(pattern.finditer(sentence))
    if matches:
        m = matches[occurrence] if occurrence < len(matches) else matches[0]
        return sentence[:m.start()] + f"{T_OPEN} {tw} {T_CLOSE}" + sentence[m.end():]
    if tw in sentence:  # fallback: first raw substring
        return sentence.replace(tw, f"{T_OPEN} {tw} {T_CLOSE}", 1)
    return f"{sentence} {T_OPEN} {tw} {T_CLOSE}"  # last resort


def build_prompt(row, candidates):
    sent = mark_target(row["sentence"], row["target_word"], row.get("target_occurrence", 0))
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        "Disambiguate the marked Uyghur word by choosing the analysis whose "
        "features are ALL correct.\n"
        f"Sentence: {sent}\n"
        f"Word: {row['target_word']}\n"
        f"Candidates:\n{cands}"
    )


import random as _random


class Seq2SeqMorphDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_source, max_target,
                 shuffle_candidates=False, fixed_shuffle=False, seed=42):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source = max_source
        self.max_target = max_target
        self.shuffle_candidates = shuffle_candidates
        self.fixed_shuffle = fixed_shuffle
        self.sources = [r.get("source", "unknown") for r in rows]

        # Fixed permutations for eval (reproducible); identity for train (reshuffled live)
        self.perms = []
        rng = _random.Random(seed)
        for r in rows:
            perm = list(range(len(r.get("candidates", []))))
            if shuffle_candidates and fixed_shuffle:
                rng.shuffle(perm)
            self.perms.append(perm)

        # Metric/baseline arrays in (possibly permuted) order
        self.gold_cand_idx, self.num_cands, self.cand_sizes = [], [], []
        for r, perm in zip(rows, self.perms):
            gold = int(r.get("label_id", -1))
            self.gold_cand_idx.append(perm.index(gold) if gold in perm else gold)
            self.num_cands.append(len(perm))
            cf = r.get("candidate_feats", [])
            self.cand_sizes.append([len(cf[p]) for p in perm] if cf else [])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        k = len(r["candidates"])
        if self.shuffle_candidates and not self.fixed_shuffle:
            perm = list(range(k))
            _random.shuffle(perm)  # dynamic per-call (train); seeded by set_seed
        else:
            perm = self.perms[idx]

        shuffled = [r["candidates"][p] for p in perm]
        gold = int(r.get("label_id", -1))
        new_gold = perm.index(gold) if gold in perm else gold
        target_text = f"REASON: {r.get('reasoning', '')} | ANSWER: {new_gold + 1}"

        enc = self.tokenizer(build_prompt(r, shuffled),
                             max_length=self.max_source, truncation=True)
        labels = self.tokenizer(text_target=target_text,
                                max_length=self.max_target, truncation=True)
        enc["labels"] = labels["input_ids"]
        return enc


# =============================================================================
# ANSWER PARSING + METRICS
# =============================================================================
def parse_answer(text):
    """Return (pred_id_1indexed | None, strict_found_bool)."""
    m = re.findall(r"ANSWER:\s*(\d+)", text)
    if m:
        return int(m[-1]), True
    m2 = re.findall(r"\d+", text)  # loose fallback
    if m2:
        return int(m2[-1]), False
    return None, False


def make_compute_metrics(tokenizer, gold_cand_idx, sources, num_cands):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, pad_id, preds)  # scrub -100 / negatives
        texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

        correct = evaluable = strict_fail = oob = 0
        per_src = defaultdict(lambda: [0, 0])
        for i, t in enumerate(texts):
            gold = gold_cand_idx[i]
            if gold < 0:
                continue
            evaluable += 1
            pid, strict = parse_answer(t)
            if not strict:
                strict_fail += 1
            hit = 0
            if pid is not None:
                rel = pid - 1
                if 0 <= rel < num_cands[i]:
                    hit = int(rel == gold)
                else:
                    oob += 1
            correct += hit
            src = sources[i]
            per_src[src][0] += hit
            per_src[src][1] += 1

        out = {
            "cand_accuracy": correct / evaluable if evaluable else 0.0,
            "n_evaluable": evaluable,
            "answer_parse_fail_rate": strict_fail / evaluable if evaluable else 0.0,
            "oob_rate": oob / evaluable if evaluable else 0.0,
        }
        for src in KNOWN_SOURCES:  # fixed keys → metric_for_best is safe
            c, tt = per_src.get(src, (0, 0))
            out[f"cand_acc_{src}"] = (c / tt) if tt else 0.0
        return out

    return compute_metrics


# =============================================================================
# TRUNCATION DIAGNOSTIC (byte-level: critical to run before training)
# =============================================================================
def truncation_report(rows, max_source, max_target):
    """Byte-level ByT5 length = len(utf-8 bytes) + 1 (EOS). No tokenizer needed."""
    src_trunc = tgt_trunc = ans_lost = 0
    for r in rows:
        s = len(build_prompt(r, r["candidates"]).encode("utf-8")) + 1
        t = len(r["target_text"].encode("utf-8")) + 1
        if s > max_source:
            src_trunc += 1
        if t > max_target:  # ANSWER is at the END → truncation kills it
            tgt_trunc += 1
            ans_lost += 1
    return src_trunc, tgt_trunc, ans_lost, len(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_rows = read_jsonl(TRAIN_FILE)
    dev_rows = read_jsonl(DEV_FILE)
    test_rows = read_jsonl(TEST_FILE)
    print(f"Train={len(train_rows)}  Dev={len(dev_rows)}  Test={len(test_rows)}")

    # ---- Truncation sanity (especially ANSWER loss on the target side) ----
    print("\n" + "=" * 50)
    print("  ✂️  TRUNCATION REPORT (byte-level)")
    print("=" * 50)
    for name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
        st, tt, al, n = truncation_report(rows, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
        print(f"  {name:5s}: src>{MAX_SOURCE_LENGTH}: {st}/{n} | "
              f"tgt>{MAX_TARGET_LENGTH}: {tt}/{n} | ANSWER-lost: {al}/{n}")
    # ANSWER-lost MUST be 0; otherwise raise MAX_TARGET_LENGTH.

    train_ds = Seq2SeqMorphDataset(train_rows, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                   shuffle_candidates=SHUFFLE_CANDIDATES, fixed_shuffle=False, seed=SEED)
    dev_ds = Seq2SeqMorphDataset(dev_rows, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                 shuffle_candidates=SHUFFLE_CANDIDATES, fixed_shuffle=True, seed=SEED)
    test_ds = Seq2SeqMorphDataset(test_rows, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                  shuffle_candidates=SHUFFLE_CANDIDATES, fixed_shuffle=True, seed=SEED)

    # ---- Model ----
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest",
        label_pad_token_id=-100, return_tensors="pt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"\nPrecision: {'bf16' if use_bf16 else 'fp32'} (fp16 disabled for T5).")

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_smoothing_factor=LABEL_SMOOTHING,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST,
        greater_is_better=GREATER_IS_BETTER,
        save_total_limit=2,
        seed=SEED,
        bf16=use_bf16,
        fp16=False,  # explicit: never fp16 for T5/ByT5
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=GEN_NUM_BEAMS,
        report_to="none",
    )

    dev_metrics = make_compute_metrics(
        tokenizer, dev_ds.gold_cand_idx, dev_ds.sources, dev_ds.num_cands)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=dev_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n✅ Best model saved to {best_dir}")

    for ckpt_dir in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, EarlyStoppingCallback)
    ]

    print("\n--- Final Evaluation: DEV (for hyperparameter selection) ---")
    trainer.compute_metrics = make_compute_metrics(
        tokenizer, dev_ds.gold_cand_idx, dev_ds.sources, dev_ds.num_cands)
    dev_metrics = trainer.evaluate(eval_dataset=dev_ds, metric_key_prefix="eval")

    print("\n--- Final Evaluation: TEST (UNSEEN, for thesis reporting only) ---")
    trainer.compute_metrics = make_compute_metrics(
        tokenizer, test_ds.gold_cand_idx, test_ds.sources, test_ds.num_cands)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    # Combine and save
    all_metrics = {**dev_metrics, **test_metrics}

    with open(os.path.join(OUTPUT_DIR, "run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
