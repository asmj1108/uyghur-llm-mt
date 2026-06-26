import os
import json
import re
import numpy as np
import torch
import shutil
import glob
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# =============================================================================
# CONFIG
# =============================================================================
def _env(key, default, cast=str):
    v = os.environ.get(key)
    if v is None:
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes")
    return cast(v)


MODEL_NAME = _env("CFG_MODEL_NAME", "xlm-roberta-large")
LABEL_VOCAB_FILE = "../label_vocab.json"
TRAIN_FILE = "../train.jsonl"
DEV_FILE = "../dev.jsonl"
TEST_FILE = "../test.jsonl"
OUTPUT_DIR = _env("CFG_OUTPUT_DIR", "xlmr-uyghur-morph-v1.1")

MAX_LENGTH = _env("CFG_MAX_LENGTH", 192, int)
SEED = _env("CFG_SEED", 42, int)

USE_TARGET_MARKERS = True
T_OPEN, T_CLOSE = "<t>", "</t>"

NUM_EPOCHS = _env("CFG_NUM_EPOCHS", 15, int)
TRAIN_BATCH_SIZE = _env("CFG_TRAIN_BATCH_SIZE", 32, int)
EVAL_BATCH_SIZE = _env("CFG_EVAL_BATCH_SIZE", 128, int)
GRAD_ACCUM_STEPS = _env("CFG_GRAD_ACCUM_STEPS", 1, int)
LEARNING_RATE = _env("CFG_LEARNING_RATE", 1e-5, float)
WEIGHT_DECAY = _env("CFG_WEIGHT_DECAY", 0.01, float)
WARMUP_RATIO = _env("CFG_WARMUP_RATIO", 0.10, float)
CLASSIFIER_DROPOUT = _env("CFG_CLASSIFIER_DROPOUT", 0.2, float)
GRAD_CHECKPOINTING = _env("CFG_GRAD_CHECKPOINTING", False, bool)

USE_LLRD = _env("CFG_USE_LLRD", False, bool)
LLRD_DECAY = _env("CFG_LLRD_DECAY", 0.95, float)

EVAL_STEPS = _env("CFG_EVAL_STEPS", 50, int)
SAVE_STEPS = _env("CFG_SAVE_STEPS", 50, int)
EARLY_STOPPING_PATIENCE = _env("CFG_EARLY_STOPPING_PATIENCE", 8, int)
METRIC_FOR_BEST = _env("CFG_METRIC_FOR_BEST", "cand_acc_llm_based")
GREATER_IS_BETTER = True

# ---- Loss for label imbalance ----
#   "bce"     : plain BCEWithLogitsLoss (HF default for multi_label)
#   "weighted": BCE with per-label pos_weight = neg/pos (clamped)
#   "focal"   : focal BCE (gamma), optionally combined with pos_weight
LOSS_TYPE = _env("CFG_LOSS_TYPE", "focal")
FOCAL_GAMMA = _env("CFG_FOCAL_GAMMA", 2.0, float)
FOCAL_USE_POS_WEIGHT = _env("CFG_FOCAL_USE_POS_WEIGHT", False, bool)

EPS = 1e-7


# =============================================================================
# LABEL VOCABULARY
# =============================================================================
def load_label_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    label2id = v["label2id"]
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, v["num_labels"]


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


def feats_to_multihot(feat_list, label2id, num_labels):
    vec = np.zeros(num_labels, dtype=np.float32)
    for feat in feat_list:
        idx = label2id.get(feat)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def feats_to_ids(feat_list, label2id):
    return [label2id[f] for f in feat_list if f in label2id]


def mark_target(sentence, target_word, occurrence=0):
    if not USE_TARGET_MARKERS:
        return sentence, target_word
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


def count_truncated_markers(rows, tokenizer, max_length):
    """How many rows lose the </t> marker due to truncation."""
    if not USE_TARGET_MARKERS:
        return 0, 0
    close_id = tokenizer.convert_tokens_to_ids(T_CLOSE)
    open_id = tokenizer.convert_tokens_to_ids(T_OPEN)
    lost = 0
    for r in rows:
        text_a, text_b = mark_target(r["sentence"], r["target_word"])
        if text_b is None:
            enc = tokenizer(text_a, truncation=True, max_length=max_length)
        else:
            enc = tokenizer(text_a, text_b, truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        # marker lost if either tag is missing after truncation
        if open_id not in ids or close_id not in ids:
            lost += 1
    return lost, len(rows)


class MorphDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, label2id, num_labels, max_length):
        self.rows = rows
        self.sources = [r.get("source", "unknown") for r in rows]
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = num_labels
        self.max_length = max_length

        self.cand_id_sets: List[List[List[int]]] = []
        self.gold_cand_idx: List[int] = []
        for r in rows:
            cand_feats = r.get("candidate_feats", [])
            self.cand_id_sets.append([feats_to_ids(cf, label2id) for cf in cand_feats])
            self.gold_cand_idx.append(int(r.get("label_id", -1)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        if USE_TARGET_MARKERS:
            text_a, text_b = mark_target(r["sentence"], r["target_word"], r.get("target_occurrence", 0))
            if text_b is None:
                enc = self.tokenizer(text_a, truncation=True, max_length=self.max_length)
            else:
                enc = self.tokenizer(text_a, text_b, truncation=True, max_length=self.max_length)
        else:
            enc = self.tokenizer(r["sentence"], r["target_word"],
                                 truncation=True, max_length=self.max_length)
        enc["labels"] = feats_to_multihot(r["label_feats"], self.label2id, self.num_labels)
        return enc


@dataclass
class MultiLabelCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]):
        labels = torch.tensor(
            np.stack([f.pop("labels") for f in features]), dtype=torch.float32
        )
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = labels
        return batch


# =============================================================================
# CANDIDATE SELECTION + METRICS
# =============================================================================
def candidate_log_likelihood(probs, cand_id_set, num_labels):
    p = np.clip(probs, EPS, 1.0 - EPS)
    in_set = np.zeros(num_labels, dtype=bool)
    in_set[cand_id_set] = True
    return np.sum(np.log(p[in_set])) + np.sum(np.log(1.0 - p[~in_set]))


def make_compute_metrics(cand_id_sets, gold_cand_idx, num_labels, sources=None):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        gold_multihot = eval_pred.label_ids
        probs = 1.0 / (1.0 + np.exp(-logits))

        # ---- threshold-0.5 diagnostics ----
        preds = (probs >= 0.5).astype(np.float32)
        tp = np.sum((preds == 1) & (gold_multihot == 1))
        fp = np.sum((preds == 1) & (gold_multihot == 0))
        fn = np.sum((preds == 0) & (gold_multihot == 1))
        micro_p = tp / (tp + fp + EPS)
        micro_r = tp / (tp + fn + EPS)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + EPS)
        exact = np.mean(np.all(preds == gold_multihot, axis=1))

        # ---- disambiguation accuracy (headline) ----
        correct = evaluable = 0
        per_src = defaultdict(lambda: [0, 0])  # src -> [correct, total]
        for i in range(len(probs)):
            cands, gold = cand_id_sets[i], gold_cand_idx[i]
            if not cands or gold < 0:
                continue
            evaluable += 1
            scores = [candidate_log_likelihood(probs[i], s, num_labels) for s in cands]
            hit = int(np.argmax(scores)) == gold
            correct += hit
            if sources is not None:
                src = sources[i]
                per_src[src][0] += hit
                per_src[src][1] += 1
        cand_acc = correct / evaluable if evaluable else 0.0
        out = {"cand_accuracy": cand_acc, "micro_f1": micro_f1,
               "exact_match": exact, "n_evaluable": evaluable}
        if sources is not None:
            for src in ("llm_based", "rule_based", "dedup"):
                c, t = per_src.get(src, (0, 0))
                out[f"cand_acc_{src}"] = (c / t) if t else 0.0
        return out

    return compute_metrics


# =============================================================================
# MODEL-FREE BASELINES
# =============================================================================
def candidate_baselines(cand_id_sets, gold_cand_idx, seed=42):
    rng = np.random.default_rng(seed)
    n = 0
    hits = {"random_expected": 0.0, "random_sampled": 0,
            "most_features": 0, "fewest_features": 0, "first": 0}
    for cands, gold in zip(cand_id_sets, gold_cand_idx):
        if not cands or gold < 0:
            continue
        n += 1
        k = len(cands)
        sizes = [len(c) for c in cands]
        hits["random_expected"] += 1.0 / k
        hits["random_sampled"] += int(rng.integers(0, k) == gold)
        hits["most_features"] += int(int(np.argmax(sizes)) == gold)
        hits["fewest_features"] += int(int(np.argmin(sizes)) == gold)
        hits["first"] += int(gold == 0)
    return {k: (v / n if n else 0.0) for k, v in hits.items()}, n


# =============================================================================
# LOSS: pos_weight + custom-loss trainer
# =============================================================================
def compute_pos_weight(rows, label2id, num_labels):
    pos = np.zeros(num_labels, dtype=np.float64)
    n = len(rows)
    for r in rows:
        for f in r["label_feats"]:
            j = label2id.get(f)
            if j is not None:
                pos[j] += 1
    neg = n - pos
    weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(weight, 1.0, 50.0), dtype=torch.float32)


class CustomLossTrainer(Trainer):
    def __init__(self, loss_type="bce", focal_gamma=2.0, pos_weight=None, **kw):
        super().__init__(**kw)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        if self.loss_type == "focal":
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none", pos_weight=pw)
            p = torch.sigmoid(logits)
            p_t = p * labels + (1 - p) * (1 - labels)
            loss = ((1 - p_t) ** self.focal_gamma * bce).mean()
        else:  # "bce" or "weighted" (weighted = bce with pw set)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pw)

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# LAYER-WISE LR DECAY
# =============================================================================
def get_llrd_optimizer(model, base_lr, decay, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = model.config.num_hidden_layers
    groups = []

    def wd(n):
        return 0.0 if any(nd in n for nd in no_decay) else weight_decay

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embeddings.word_embeddings" in n:
            lr = base_lr  # full LR: the new marker rows must move
        elif "embeddings" in n:
            lr = base_lr * (decay ** (num_layers + 1))
        elif "encoder.layer." in n:
            layer = int(n.split("encoder.layer.")[1].split(".")[0])
            lr = base_lr * (decay ** (num_layers - layer))
        else:
            lr = base_lr
        groups.append({"params": [p], "lr": lr, "weight_decay": wd(n)})
    return AdamW(groups)


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label2id, id2label, num_labels = load_label_vocab(LABEL_VOCAB_FILE)
    print(f"Loaded {num_labels} feature labels.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_rows = read_jsonl(TRAIN_FILE)
    dev_rows = read_jsonl(DEV_FILE)
    test_rows = read_jsonl(TEST_FILE)
    print(f"Train={len(train_rows)}  Dev={len(dev_rows)}  Test={len(test_rows)}")

    train_ds = MorphDataset(train_rows, tokenizer, label2id, num_labels, MAX_LENGTH)
    dev_ds = MorphDataset(dev_rows, tokenizer, label2id, num_labels, MAX_LENGTH)
    test_ds = MorphDataset(test_rows, tokenizer, label2id, num_labels, MAX_LENGTH)

    # ---- Baselines (model-free, report once) ----
    print("\n" + "=" * 50)
    print("  📊 MODEL-FREE CANDIDATE BASELINES")
    print("=" * 50)
    for split_name, ds in [("dev", dev_ds), ("test", test_ds)]:
        base, n = candidate_baselines(ds.cand_id_sets, ds.gold_cand_idx, seed=SEED)
        print(f"\n  {split_name} (n={n} evaluable):")
        for k, v in base.items():
            print(f"    {k:18s}: {v:.4f}")

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=CLASSIFIER_DROPOUT,
        id2label=id2label,
        label2id=label2id,
    )

    print("classifier_dropout =", model.config.classifier_dropout)

    if USE_TARGET_MARKERS:
        tokenizer.add_special_tokens({"additional_special_tokens": [T_OPEN, T_CLOSE]})
        model.resize_token_embeddings(len(tokenizer))
        # Initialize the two new rows to the mean of existing embeddings (far better than random)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            n_new = 2
            emb[-n_new:] = emb[:-n_new].mean(dim=0, keepdim=True)

        for name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
            lost, total = count_truncated_markers(rows, tokenizer, MAX_LENGTH)
            pct = 100 * lost / total if total else 0
            print(f"⚠️  {name}: {lost}/{total} rows ({pct:.2f}%) lose a target marker at MAX_LENGTH={MAX_LENGTH}")

    if GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    collator = MultiLabelCollator(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
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
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )

    # ---- pos_weight (for weighted / focal+pos_weight) ----
    pos_weight = None
    if LOSS_TYPE == "weighted" or (LOSS_TYPE == "focal" and FOCAL_USE_POS_WEIGHT):
        pos_weight = compute_pos_weight(train_rows, label2id, num_labels)

    # ---- LLRD optimizer (built after embedding resize) ----
    optimizer = None
    if USE_LLRD:
        optimizer = get_llrd_optimizer(model, LEARNING_RATE, LLRD_DECAY, WEIGHT_DECAY)
        for g in optimizer.param_groups:
            for p in g["params"]:
                # crude: find the big [vocab, hidden] tensor
                if p.shape[0] == len(tokenizer):
                    print("word_embeddings LR =", g["lr"])

    dev_metrics = make_compute_metrics(dev_ds.cand_id_sets, dev_ds.gold_cand_idx, num_labels, sources=dev_ds.sources)

    trainer = CustomLossTrainer(
        loss_type=LOSS_TYPE,
        focal_gamma=FOCAL_GAMMA,
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=dev_metrics,
        optimizers=(optimizer, None),  # scheduler auto-created from args
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # ---- Train ----
    trainer.train()

    # ---- Save best ----
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n✅ Best model saved to {best_dir}")

    # ---- CLEANUP INTERMEDIATE CHECKPOINTS ----
    print("\n🧹 Cleaning up intermediate checkpoints...")
    for ckpt_dir in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    # Detach early-stopping callback so it doesn't warn
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, EarlyStoppingCallback)
    ]

    print("\n--- Final Evaluation: DEV (for hyperparameter selection) ---")
    trainer.compute_metrics = make_compute_metrics(
        dev_ds.cand_id_sets, dev_ds.gold_cand_idx, num_labels, sources=dev_ds.sources)
    dev_metrics = trainer.evaluate(eval_dataset=dev_ds, metric_key_prefix="eval")

    print("\n--- Final Evaluation: TEST (UNSEEN, for thesis reporting only) ---")
    trainer.compute_metrics = make_compute_metrics(
        test_ds.cand_id_sets, test_ds.gold_cand_idx, num_labels, sources=test_ds.sources)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    # Combine and save
    all_metrics = {**dev_metrics, **test_metrics}

    with open(os.path.join(OUTPUT_DIR, "run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Dump the config so we know what generated these metrics
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "seed": SEED, "lr": LEARNING_RATE, "use_llrd": USE_LLRD,
            "llrd_decay": LLRD_DECAY, "loss_type": LOSS_TYPE,
            "focal_pos_weight": FOCAL_USE_POS_WEIGHT,
            "train_batch": TRAIN_BATCH_SIZE, "warmup": WARMUP_RATIO,
        }, f, indent=2)


if __name__ == "__main__":
    main()
