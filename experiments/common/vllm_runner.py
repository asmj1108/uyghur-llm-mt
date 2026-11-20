"""
vllm_runner.py - Direct (in-process) vLLM batch inference for the MT experiments.

- No server: `vllm.LLM` is instantiated directly in the driver script.
- Qwen3.5 thinking mode is force-disabled via the chat template
  (`enable_thinking=False` pre-fills `<think>\\n\\n</think>\\n\\n`, so the model
  emits the translation directly and cannot disappear into a reasoning loop).
- Greedy decoding (temperature=0) for reproducible thesis numbers.
"""

import gc
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams

# Qwen3.5 chat templates accept this kwarg; pre-fills an empty <think> block.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def make_llm(
    model_id: str,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
    seed: int = 0,
    enforce_eager: bool = False,
) -> LLM:
    """Instantiate a text-generation vLLM engine for one model.

    enforce_eager=True disables torch.compile + CUDA graphs; needed for
    architectures whose graphs hit inductor bugs (Mistral-3 family) —
    slower decode, identical greedy outputs.
    """
    return LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        trust_remote_code=True,
    )


def free_llm(llm: Optional[LLM]) -> None:
    """Release engine memory between models (esp. before the 27B load)."""
    if llm is None:
        return
    try:
        if hasattr(llm, "llm_engine"):
            del llm.llm_engine
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _strip_think(text: str) -> str:
    """Defensive cleanup: drop any leaked reasoning block."""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _split_thinking(raw: str) -> tuple[str, str, bool]:
    """Split a thinking-mode generation into (reasoning, answer, answered).

    With enable_thinking=True the chat template pre-fills '<think>\\n', so the
    model emits reasoning first and closes it with '</think>'. If the tag never
    arrives (reasoning loop / context exhaustion) the run produced no answer.
    """
    if "</think>" in raw:
        reasoning, answer = raw.split("</think>", 1)
        return reasoning.strip(), answer.strip(), True
    return raw.strip(), "", False


def _split_thinking_marker(raw: str, marker: str) -> tuple[str, str, bool]:
    """Split a thinking-mode generation on a model-specific channel marker.

    Gemma-4 with enable_thinking=True emits reasoning first, then the
    channel marker '<channel|>', then the content — the same convention as
    agent_loop.THINK_MARKER_BY_MODEL in the Exp4 agent loop.
    """
    if marker in raw:
        reasoning, answer = raw.split(marker, 1)
        return reasoning.strip(), answer.strip(), True
    return raw.strip(), "", False


def chat_generate(
    llm: LLM,
    user_prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    top_p: float = 1.0,
    top_k: int = -1,
    think_marker: str = None,
) -> List[Dict]:
    """Batch-generate one completion per prompt via the model's chat template.

    With enable_thinking=True, the returned dicts additionally carry
    'reasoning' (raw reasoning text, for loop analysis) and 'answered'
    (False = never closed the think block within max_tokens). The split
    convention is model-specific: Qwen closes its pre-filled thinking block
    with the standard closing tag (default); Gemma-4 delimits its thought
    channel with '<channel|>' — pass think_marker for that family.

    top_p/top_k allow vendor-recommended thinking-mode sampling
    (temperature=0.6, top_p=0.95, top_k=20 for Qwen3 thinking).

    Returns a list of dicts: {text, reasoning, answered, prompt_tokens,
    completion_tokens, finish_reason}.
    """
    conversations = [[{"role": "user", "content": p}] for p in user_prompts]
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        use_tqdm=True,
    )

    results = []
    for out in outputs:
        raw = out.outputs[0].text
        if enable_thinking:
            if think_marker:
                reasoning, text, answered = _split_thinking_marker(raw, think_marker)
            else:
                reasoning, text, answered = _split_thinking(raw)
        else:
            reasoning, answered = None, True
            text = _strip_think(raw)
        n_prompt = len(out.prompt_token_ids) if out.prompt_token_ids else 0
        n_completion = len(out.outputs[0].token_ids)
        finish_reason = out.outputs[0].finish_reason
        results.append({
            "text": text,
            "reasoning": reasoning,
            "answered": answered,
            "prompt_tokens": n_prompt,
            "completion_tokens": n_completion,
            "finish_reason": finish_reason,  # 'stop' | 'length' | ...
        })
    return results
