import os
import re
import json
import random as _random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# =============================================================================
# CONFIG
# =============================================================================
TEST_FILE = "dev.jsonl"
SEED = 42
BATCH = 32

# mode: "cot" | "direct" | "multitask"; type: "seq2seq" | "causal"
LLMS = [
    {"name": "byt5-base", "path": "byt5-base-multitask-1e-4/best",      "type": "seq2seq", "mode": "multitask"},
    {"name": "qwen-0.8b", "path": "qwen-cot-2e-5/best", "type": "causal",  "mode": "cot"},
    {"name": "qwen-2b",   "path": "qwen2b-direct-9e-6/best",   "type": "causal",  "mode": "direct"},
]

T_OPEN, T_CLOSE = "<t>", "</t>"


# =============================================================================
# SHARED HELPERS
# =============================================================================
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mark_target(sentence, target_word, occurrence=0):
    tw = target_word.strip()
    if not tw:
        return sentence
    pat = re.compile(rf"(?<!\S){re.escape(tw)}(?!\S)")
    matches = list(pat.finditer(sentence))
    if matches:
        m = matches[occurrence] if occurrence < len(matches) else matches[0]
        return sentence[:m.start()] + f"{T_OPEN} {tw} {T_CLOSE}" + sentence[m.end():]
    if tw in sentence:
        return sentence.replace(tw, f"{T_OPEN} {tw} {T_CLOSE}", 1)
    return f"{sentence} {T_OPEN} {tw} {T_CLOSE}"


def build_prompt_seq2seq(row, cands):
    sent = mark_target(row["sentence"], row["target_word"], row.get("target_occurrence", 0))
    c = "\n".join(f"{i+1}. {x}" for i, x in enumerate(cands))
    return ("Disambiguate the marked Uyghur word by choosing the analysis whose "
            "features are ALL correct.\n"
            f"Sentence: {sent}\nWord: {row['target_word']}\nCandidates:\n{c}")


def build_prompt_causal(row, cands):
    sent = mark_target(row["sentence"], row["target_word"], row.get("target_occurrence", 0))
    c = "\n".join(f"{i+1}. {x}" for i, x in enumerate(cands))
    return ("You are an expert computational linguist specializing in Uyghur morphosyntax "
            "and contextual disambiguation.\n"
            "Disambiguate the marked Uyghur word by choosing the analysis whose features "
            "are ALL correct in context.\n"
            f"Sentence: {sent}\nWord: {row['target_word']}\nCandidates:\n{c}")


def parse_answer(text):
    m = re.findall(r"ANSWER:\s*(\d+)", text)
    if m:
        return int(m[-1])
    m2 = re.findall(r"\d+", text)
    return int(m2[-1]) if m2 else None


def fixed_perms(rows, seed=SEED):
    """Reproduce the eval-time fixed shuffle used by analysis scripts."""
    rng = _random.Random(seed)
    perms = []
    for r in rows:
        perm = list(range(len(r.get("candidates", []))))
        rng.shuffle(perm)
        perms.append(perm)
    return perms


# =============================================================================
# INFERENCE
# =============================================================================
@torch.inference_mode()
def run_seq2seq(cfg, rows, perms):
    tok = AutoTokenizer.frompretrained(cfg["path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["path"]).cuda().eval()
    max_new = 448 if cfg["mode"] != "direct" else 16
    preds = []
    for i in range(0, len(rows), BATCH):
        batch, bperm = rows[i:i+BATCH], perms[i:i+BATCH]
        texts = []
        for r, p in zip(batch, bperm):
            cands = [r["candidates"][j] for j in p]
            t = build_prompt_seq2seq(r, cands)
            if cfg["mode"] == "multitask":
                t = "[Direct Answer] " + t
            texts.append(t)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2560).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False)
        preds += tok.batch_decode(out, skip_special_tokens=True)
    del model; torch.cuda.empty_cache()
    return preds


@torch.inference_mode()
def run_causal(cfg, rows, perms):
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    tok.padding_side = "left"
    dtype = torch.float32 if cfg["mode"] == "cot" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=dtype, attn_implementation="sdpa").cuda().eval()
    model.config.use_cache = True
    max_new = 6 if cfg["mode"] in ("direct", "multitask") else 200
    preds = []
    for i in range(0, len(rows), BATCH):
        batch, bperm = rows[i:i+BATCH], perms[i:i+BATCH]
        prompts = []
        for r, p in zip(batch, bperm):
            cands = [r["candidates"][j] for j in p]
            base = build_prompt_causal(r, cands)
            if cfg["mode"] == "multitask":
                base = "[Direct Answer]\n" + base
            ids = tok.apply_chat_template([{"role": "user", "content": base}],
                                          tokenize=False, add_generation_prompt=True)
            prompts.append(ids)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        preds += tok.batch_decode(gen, skip_special_tokens=True)
    del model; torch.cuda.empty_cache()
    return preds


def main():
    rows = read_jsonl(TEST_FILE)
    perms = fixed_perms(rows)
    num_cands = [len(p) for p in perms]

    for cfg in LLMS:
        print(f"\n>>> Running generation for: {cfg['name']}")
        raw = run_seq2seq(cfg, rows, perms) if cfg["type"] == "seq2seq" else run_causal(cfg, rows, perms)

        preds_dict = {}
        for row_idx, (t, nc) in enumerate(zip(raw, num_cands)):
            pid = parse_answer(t)
            # convert from 1-indexed (prompt) to 0-indexed relative idx
            rel_idx = pid - 1 if pid is not None and 0 <= pid - 1 < nc else None
            preds_dict[str(row_idx)] = rel_idx

        out_name = f"{cfg['name']}_dev_preds.json"
        with open(out_name, "w") as f:
            json.dump(preds_dict, f, indent=2)
        print(f"Saved {out_name}")

if __name__ == "__main__":
    main()