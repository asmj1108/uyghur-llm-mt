#!/usr/bin/env python3
"""
run_exp5.py - Experiment 5: Thinking mode x C3L (the lean optimum).

Does native thinking substitute for, complement, or interfere with the pushed
lemma-keyed glosses? 2x2(+1) per model over FLORES dev (997):

  arm A   greedy    C0, C3L  — reuse (Exp1/2/3) except Qwen-27B C3L (new)
  arm S   offrec    C0, C3L  — thinking OFF + recommended sampling (new)
  arm T   thinkrec  C0, C3L  — thinking ON + recommended sampling (Qwen C0
                                reused from Exp2b; rest new)

See DESIGN.md for hypotheses (substitution / complementarity / interference)
and the deconfounding logic (T vs S isolates thinking; S vs A sanity-checks
the decoding change).

Usage (vLLM env):
  # smoke test (Gemma static thinking validation, 3 episodes)
  "$TMPDIR/vllm/bin/python" -u experiments/exp5_thinking_static/run_exp5.py \
      --phase translate --arms thinkrec --models google/gemma-4-E4B-it --limit 3
  # full run (all models, all arms; resumes via checkpoints)
  "$TMPDIR/vllm/bin/python" -u experiments/exp5_thinking_static/run_exp5.py \
      --phase translate
"""

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from experiments.common import checkpoint as ckpt  # noqa: E402
from experiments.common import vllm_runner  # noqa: E402
from experiments.common.prompt_variants import (  # noqa: E402
    build_prompts_for_condition, get_condition, load_flores_records,
    load_references,
)
from experiments.exp5_thinking_static.cells import (  # noqa: E402
    ARMS, CONDITIONS, DEV_REF, FLORES_RESULTS_PATH, MODELS, cell_path,
    thinking_stats,
)

# Stop-rule gate inherited from Exp2b: below this answered rate a thinking
# cell is a finding (reported), not a reason to silently continue.
MIN_ANSWERED_RATE = 0.50


def parse_args():
    ap = argparse.ArgumentParser(description="Exp5: thinking x C3L")
    ap.add_argument("--phase", choices=["translate", "report"], default="translate",
                    help="translate: generate; report: answered-rate/loop report only")
    ap.add_argument("--models", nargs="+",
                    default=[m for m, _, _ in MODELS])
    ap.add_argument("--arms", nargs="+", choices=list(ARMS),
                    default=list(ARMS))
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=64)
    return ap.parse_args()


def run_cells(args, model_id, tag, records, refs):
    n = len(records)
    items = [{"id": i, "sentence": rec["sentence"], "reference": refs[i]}
             for i, rec in enumerate(records)]

    llm = vllm_runner.make_llm(model_id,
                                gpu_memory_utilization=args.gpu_mem_util,
                                max_model_len=args.max_model_len)
    t0 = time.time()
    try:
        for arm in args.arms:
            enable_thinking, sampling, max_tokens = ARMS[arm]
            for cond_name in args.conditions:
                cond = get_condition(cond_name)
                path = cell_path(tag, cond_name, arm, args.limit)
                completed = ckpt.load_completed(path)
                todo = ckpt.pending_items(items, completed)
                status = (f"[exp5] {model_id} {cond.cid} [{arm}]: "
                          f"{len(completed)}/{n} done"
                          + (" — skipping (complete)" if not todo
                             else f" — generating {len(todo)}"))
                print(status, flush=True)
                if not todo:
                    continue

                pending_ids = {it["id"] for it in todo}
                prompts_all = build_prompts_for_condition(cond, records)
                id_to_prompt = {i: prompts_all[i] for i in pending_ids}

                writer = ckpt.JSONLCheckpointWriter(path)
                try:
                    for start in range(0, len(todo), args.chunk_size):
                        chunk = todo[start:start + args.chunk_size]
                        gens = vllm_runner.chat_generate(
                            llm,
                            [id_to_prompt[it["id"]] for it in chunk],
                            max_tokens=max_tokens,
                            enable_thinking=enable_thinking,
                            **sampling,
                        )
                        for it, gen in zip(chunk, gens):
                            rec = ckpt.new_record(
                                it["id"], model_id, f"{cond.cid}_{arm}",
                                it["sentence"], it["reference"],
                                id_to_prompt[it["id"]],
                                thinking=enable_thinking)
                            writer.write(ckpt.finalize_record(rec, gen))
                        done = len(completed) + min(start + args.chunk_size,
                                                    len(todo))
                        print(f"[exp5]   {cond.cid} [{arm}]: {done}/{n}",
                              flush=True)
                finally:
                    writer.close()
    finally:
        vllm_runner.free_llm(llm)
    print(f"[exp5] {model_id} finished in {time.time() - t0:.1f}s "
          f"(memory freed)", flush=True)


def report(args):
    """Answered-rate / loop report for every cell of every requested model."""
    tag_by_id = {m: t for m, t, _ in MODELS}
    for model_id in args.models:
        tag = tag_by_id[model_id]
        for arm in args.arms:
            for cond_name in args.conditions:
                path = cell_path(tag, cond_name, arm, args.limit)
                st = thinking_stats(path)
                if not st["n"]:
                    print(f"[exp5] {model_id} {cond_name} [{arm}]: no data")
                    continue
                rate = st["answered_rate"]
                flag = ""
                if arm == "thinkrec" and (rate or 0) < MIN_ANSWERED_RATE:
                    flag = "  !! BELOW STOP-RULE GATE (findings, see Exp2b)"
                print(f"[exp5] {model_id} {cond_name} [{arm}]: "
                      f"answered {100 * (rate or 0):.1f}% "
                      f"(loops {st['looping_reasoning']}, "
                      f"len-cap {st['length_capped']}){flag}", flush=True)


def main():
    args = parse_args()
    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs)
    if args.limit:
        records, refs = records[:args.limit], refs[:args.limit]
    print(f"[exp5] {len(records)} sentences | arms: {args.arms} | "
          f"models: {args.models}", flush=True)

    if args.phase == "report":
        report(args)
        return

    tag_by_id = {m: t for m, t, _ in MODELS}
    for model_id in args.models:
        run_cells(args, model_id, tag_by_id[model_id], records, refs)
    report(args)


if __name__ == "__main__":
    main()
