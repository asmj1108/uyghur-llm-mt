import os
import re
import json
import glob
import shutil
import random as _random
from collections import defaultdict

import numpy as np
import torch
from torch.ao.quantization.utils import weight_dtype
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithFlattening,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# =============================================================================
#  CONFIG
# =============================================================================
def _env(key, default, cast=str):
    v = os.environ.get(key)
    if v is None:
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes")
    return cast(v)


# Options: "cot" | "direct" | "multitask"
TRAIN_MODE = _env("CFG_TRAIN_MODE", "cot")
MODEL_NAME = _env("CFG_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
TRAIN_FILE = "../train.jsonl"
DEV_FILE = "../dev.jsonl"
TEST_FILE = "../test.jsonl"
OUTPUT_DIR = _env("CFG_OUTPUT_DIR", "qwen-0.8b-cot-2e-5")

SHUFFLE_CANDIDATES = True
SEED = _env("CFG_SEED", 42, int)

USE_PACKING = _env("CFG_USE_PACKING", True, bool)

# ---- Lengths ----
MAX_SOURCE_LENGTH = _env("CFG_MAX_SOURCE_LENGTH", 1024, int)  # prompt tokens
MAX_TARGET_LENGTH = _env("CFG_MAX_TARGET_LENGTH", 256, int)  # target tokens

# ---- Target-word grounding ----
USE_TARGET_MARKERS = True
T_OPEN, T_CLOSE = "<t>", "</t>"

# ---- Hyperparameters (decoder-only LMs want LOW lr) ----
NUM_EPOCHS = _env("CFG_NUM_EPOCHS", 30, int)
TRAIN_BATCH_SIZE = _env("CFG_TRAIN_BATCH_SIZE", 1, int)
EVAL_BATCH_SIZE = _env("CFG_EVAL_BATCH_SIZE", 16, int)
GRAD_ACCUM_STEPS = _env("CFG_GRAD_ACCUM_STEPS", 1, int)
LEARNING_RATE = _env("CFG_LR", 2e-5, float)
WEIGHT_DECAY = _env("CFG_WEIGHT_DECAY", 0.0, float)
WARMUP_RATIO = _env("CFG_WARMUP_RATIO", 0.10, float)
GRAD_CHECKPOINTING = _env("CFG_GRAD_CHECKPOINTING", False, bool)

# ---- Eval / generation / early stopping ----
EVAL_STEPS = _env("CFG_EVAL_STEPS", 300, int)
SAVE_STEPS = _env("CFG_SAVE_STEPS", 300, int)
EARLY_STOPPING_PATIENCE = _env("CFG_EARLY_STOPPING_PATIENCE", 8, int)
METRIC_FOR_BEST = "cand_acc_llm_based"
GREATER_IS_BETTER = True

KNOWN_SOURCES = ("llm_based", "rule_based", "dedup")

# During eval, direct/multitask emit a bare answer; cot emits reason+answer.
EVAL_IS_DIRECT = TRAIN_MODE in ("direct", "multitask")
# len(tokenizer("ANSWER: 9")["input_ids"] = 5
GEN_MAX_NEW_TOKENS = 6 if EVAL_IS_DIRECT else 200


# =============================================================================
# DATA HELPERS
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
    if tw in sentence:
        return sentence.replace(tw, f"{T_OPEN} {tw} {T_CLOSE}", 1)
    return f"{sentence} {T_OPEN} {tw} {T_CLOSE}"


def build_task_text(row, candidates):
    """The user-turn content (before chat templating)."""
    sent = mark_target(row["sentence"], row["target_word"], row.get("target_occurrence", 0))
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        "You are an expert computational linguist specializing in Uyghur morphosyntax and contextual disambiguation.\n"
        "Disambiguate the marked Uyghur word by choosing the analysis whose features are ALL correct in context.\n"
        f"Sentence: {sent}\n"
        f"Word: {row['target_word']}\n"
        f"Candidates:\n{cands}"
    )


def make_io(row, shuffled, new_gold, mode, is_eval, rng):
    """Return (user_content, target_text). target_text unused at eval time."""
    base = build_task_text(row, shuffled)
    reasoning = row.get("reasoning", "")
    ans = new_gold + 1

    if mode == "cot":
        return base, f"REASON: {reasoning} | ANSWER: {ans}"

    if mode == "direct":
        return base, f"ANSWER: {ans}"

    # multitask
    if is_eval:
        return "[Direct Answer]\n" + base, f"ANSWER: {ans}"
    if rng.random() < 0.5:
        return "[Reasoning]\n" + base, f"REASON: {reasoning} | ANSWER: {ans}"
    return "[Direct Answer]\n" + base, f"ANSWER: {ans}"


def apply_user_template(tokenizer, content):
    """Render to string first to avoid C++ object bleed, then tokenize. Thinking Mode disabled on default"""
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)["input_ids"]
    # return tokenizer(rendered_str, add_special_tokens=False)["input_ids"]


# =============================================================================
# DATASET
# =============================================================================
class CausalMorphDataset(Dataset):
    def __init__(self, rows, tokenizer, mode, max_source, max_target,
                 is_eval, shuffle_candidates, seed=42):
        self.rows = rows
        self.tok = tokenizer
        self.mode = mode
        self.max_source = max_source
        self.max_target = max_target
        self.is_eval = is_eval
        self.shuffle_candidates = shuffle_candidates
        self.eos_id = tokenizer.eos_token_id
        self.sources = [r.get("source", "unknown") for r in rows]
        self._rng = _random.Random(seed)

        # Fixed permutations for eval (reproducible); live reshuffle for train.
        self.perms = []
        rng = _random.Random(seed)
        for r in rows:
            perm = list(range(len(r.get("candidates", []))))
            if shuffle_candidates and is_eval:
                rng.shuffle(perm)
            self.perms.append(perm)

        # Order-based gold arrays for compute_metrics (eval order == dataset order).
        self.gold_cand_idx, self.num_cands = [], []
        for r, perm in zip(rows, self.perms):
            gold = int(r.get("label_id", -1))
            self.gold_cand_idx.append(perm.index(gold) if gold in perm else gold)
            self.num_cands.append(len(perm))

        self._eager = None
        if is_eval:
            self._eager = [self._build(i) for i in range(len(self.rows))]

    def __len__(self):
        return len(self.rows)

    def _perm_for(self, idx, k):
        if self.shuffle_candidates and not self.is_eval:
            perm = list(range(k))
            self._rng.shuffle(perm)
            return perm
        return self.perms[idx]

    def __getitem__(self, idx):
        return self._eager[idx] if self._eager is not None else self._build(idx)

    def _build(self, idx):
        r = self.rows[idx]
        k = len(r["candidates"])
        perm = self._perm_for(idx, k)
        shuffled = [r["candidates"][p] for p in perm]
        gold = int(r.get("label_id", -1))
        new_gold = perm.index(gold) if gold in perm else gold

        user, target = make_io(r, shuffled, new_gold, self.mode, self.is_eval, self._rng)

        prompt_ids = apply_user_template(self.tok, user)
        # Left-truncate the prompt: keep the trailing generation marker + candidates.
        if len(prompt_ids) > self.max_source:
            prompt_ids = prompt_ids[-self.max_source:]

        target_ids = self.tok(target, add_special_tokens=False)["input_ids"] + [self.eos_id]
        # Keep the END of the target (the ANSWER lives at the end).
        if len(target_ids) > self.max_target:
            target_ids = target_ids[-self.max_target:]

        if self.is_eval:
            # Prompt only for generation; target kept just for label-presence.
            return {"input_ids": prompt_ids, "labels": target_ids}

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids  # completion-only loss
        return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# COLLATORS
# =============================================================================
def make_train_collator(pad_id):
    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)  # right pad
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def make_eval_collator(pad_id):
    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn = [], []
        for f in features:
            ids = f["input_ids"]
            pad = maxlen - len(ids)
            input_ids.append([pad_id] * pad + ids)  # LEFT pad for generation
            attn.append([0] * pad + [1] * len(ids))
        maxl = max(len(f["labels"]) for f in features)  # presence only
        labels = [f["labels"] + [-100] * (maxl - len(f["labels"])) for f in features]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


# =============================================================================
# ANSWER PARSING + METRICS
# =============================================================================
def parse_answer(text):
    m = re.findall(r"ANSWER:\s*(\d+)", text)
    if m:
        return int(m[-1]), True
    m2 = re.findall(r"\d+", text)
    if m2:
        return int(m2[-1]), False
    return None, False


def make_compute_metrics(tokenizer, gold_cand_idx, sources, num_cands):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, pad_id, preds)  # scrub -100 padding
        texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(texts[0])

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
        for src in KNOWN_SOURCES:
            c, tt = per_src.get(src, (0, 0))
            out[f"cand_acc_{src}"] = (c / tt) if tt else 0.0
        return out

    return compute_metrics


# =============================================================================
# CUSTOM TRAINER: generation-based eval for a causal LM
# =============================================================================
class CausalGenTrainer(Trainer):
    def __init__(self, *args, gen_max_new_tokens=64, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_max_new_tokens = gen_max_new_tokens
        self._eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self._eval_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluation_loop(self, *args, **kwargs):
        m = self.model
        orig_impl, orig_cache = m.config._attn_implementation, m.config.use_cache
        m.set_attn_implementation("sdpa")   # FA2 needs fp16/bf16; eval runs fp32
        m.config.use_cache = True
        try:
            return super().evaluation_loop(*args, **kwargs)
        finally:
            m.set_attn_implementation(orig_impl)
            m.config.use_cache = orig_cache

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )
        return (None, generated[:, input_ids.shape[1]:], inputs.get("labels"))


# =============================================================================
# TRUNCATION DIAGNOSTIC (token-level)
# =============================================================================
def truncation_report(rows, tokenizer, mode, max_source, max_target):
    src_trunc = tgt_trunc = ans_lost = 0
    rng = _random.Random(0)
    for r in rows:
        perm = list(range(len(r["candidates"])))
        shuffled = [r["candidates"][p] for p in perm]
        gold = int(r.get("label_id", -1))
        new_gold = perm.index(gold) if gold in perm else gold
        user, target = make_io(r, shuffled, new_gold, mode, is_eval=False, rng=rng)
        p = len(apply_user_template(tokenizer, user))
        t = len(tokenizer(target, add_special_tokens=False)["input_ids"]) + 1
        if p > max_source:
            src_trunc += 1
        if t > max_target:
            tgt_trunc += 1
            ans_lost += 1  # ANSWER is at the end -> left-keep saves it, but flag anyway
    return src_trunc, tgt_trunc, ans_lost, len(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"

    train_rows = read_jsonl(TRAIN_FILE)
    dev_rows = read_jsonl(DEV_FILE)
    test_rows = read_jsonl(TEST_FILE)
    print(f"Mode={TRAIN_MODE}  Model={MODEL_NAME}")
    print(f"Train={len(train_rows)}  Dev={len(dev_rows)}  Test={len(test_rows)}")

    if _env("CFG_TRUNC_REPORT", False, bool):
        print("\n" + "=" * 50)
        print("  ✂️  TRUNCATION REPORT (token-level)")
        print("=" * 50)
        for name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
            st, tt, al, n = truncation_report(rows, tokenizer, TRAIN_MODE,
                                              MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
            print(f"  {name:5s}: src>{MAX_SOURCE_LENGTH}: {st}/{n} | "
                  f"tgt>{MAX_TARGET_LENGTH}: {tt}/{n} | ANSWER-at-risk: {al}/{n}")

    train_ds = CausalMorphDataset(train_rows, tokenizer, TRAIN_MODE,
                                  MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                  is_eval=False, shuffle_candidates=SHUFFLE_CANDIDATES, seed=SEED)
    dev_ds = CausalMorphDataset(dev_rows, tokenizer, TRAIN_MODE,
                                MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                is_eval=True, shuffle_candidates=SHUFFLE_CANDIDATES, seed=SEED)
    test_ds = CausalMorphDataset(test_rows, tokenizer, TRAIN_MODE,
                                 MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH,
                                 is_eval=True, shuffle_candidates=SHUFFLE_CANDIDATES, seed=SEED)

    use_bf16 = True if USE_PACKING else torch.cuda.is_bf16_supported()
    print(f"\nTraining Precision: {'bf16' if use_bf16 else 'fp32'}")

    if torch.cuda.is_tf32_supported():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 turned on for generation")

    # bfloat16 unstable as generation precision if max_new_token is above 6, e.g., in CoT mode
    # with left-padded batched inputs, due to linear attention of the Qwen3.5 model
    _weight_dtype = torch.float32 if TRAIN_MODE == "cot" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=_weight_dtype,
        attn_implementation="flash_attention_2" if USE_PACKING else "sdpa",
    )
    print("Model weight loaded in ", str(_weight_dtype))

    if GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if USE_PACKING:
        train_collator = DataCollatorWithFlattening(
            return_flash_attn_kwargs=True,   # cu_seq_lens_q/k, max_length_q/k -> FLA + FA2
            return_seq_idx=True,             # seq_idx -> causal conv1d
        )
    else:
        # right-padding fallback
        train_collator = make_train_collator(tokenizer.pad_token_id)
    eval_collator = make_eval_collator(tokenizer.pad_token_id)

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
        bf16=use_bf16,  # forward/backward in bf16 autocast
        fp16=False,
        gradient_checkpointing=GRAD_CHECKPOINTING,
        report_to="none",
        remove_unused_columns=False, # just to be safe
    )

    dev_metrics_fn = make_compute_metrics(
        tokenizer, dev_ds.gold_cand_idx, dev_ds.sources, dev_ds.num_cands)

    trainer = CausalGenTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=train_collator,
        compute_metrics=dev_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        eval_collator=eval_collator,
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

    print("\n--- Final Evaluation: DEV (hyperparameter selection) ---")
    trainer.compute_metrics = make_compute_metrics(
        tokenizer, dev_ds.gold_cand_idx, dev_ds.sources, dev_ds.num_cands)
    dev_metrics = trainer.evaluate(eval_dataset=dev_ds, metric_key_prefix="eval")

    print("\n--- Final Evaluation: TEST (UNSEEN, thesis reporting only) ---")
    trainer.compute_metrics = make_compute_metrics(
        tokenizer, test_ds.gold_cand_idx, test_ds.sources, test_ds.num_cands)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    all_metrics = {**dev_metrics, **test_metrics}
    with open(os.path.join(OUTPUT_DIR, "run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_mode": TRAIN_MODE,
            "lr": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "patience": EARLY_STOPPING_PATIENCE,
            "eval_steps": EVAL_STEPS,
            "save_steps": SAVE_STEPS,
        }, f, indent=2)


if __name__ == "__main__":
    main()
