"""
agent_loop.py - Exp 4: round-synchronous agentic retrieval loop.

Each sentence is an independent single-agent episode:
  user prompt (C0-style + tool notice) -> [assistant tool calls -> tool
  results]* -> final translation.
All still-active episodes are batched into ONE llm.chat() call per round.

Tool calls are parsed with vLLM's own per-family parsers via
ToolParserRegistry (qwen3_coder / llama3_json / mistral / gemma4).

Runs in the vLLM env. Every turn, call, and return is logged per episode
(JSONL, flush-per-episode, resumable at episode granularity).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from experiments.common.vllm_runner import _strip_think

# registry names, per family
PARSER_BY_MODEL = {
    "Qwen/Qwen3.5-9B": "qwen3_coder",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_json",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "mistral",
    "google/gemma-4-E4B-it": "gemma4",
    "google/gemma-4-12b-it": "gemma4",   # same family/parser as E4B
}

# Qwen3.5: thinking force-disabled (same regime as every prior experiment)
CHAT_KWARGS_BY_MODEL = {
    "Qwen/Qwen3.5-9B": {"enable_thinking": False},
}

# Thinking-output markers: text before the marker is reasoning, after is
# content (tool calls / final answer)
THINK_MARKER_BY_MODEL = {
    "Qwen/Qwen3.5-9B": "**END**",
    "google/gemma-4-E4B-it": "<channel|>",
    "google/gemma-4-12b-it": "<channel|>",  # same Gemma-4 thinking channel
}

# Llama-3.1's chat template only supports a single tool call per assistant
# message (raises VLLMValidationError otherwise). For such models we execute
# only the first call of a turn and log the rest as skipped.
SINGLE_CALL_PER_TURN = {"meta-llama/Llama-3.1-8B-Instruct"}

# Qwen's official thinking-mode sampling (also used for Gemma's thinking arm)
RECOMMENDED_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

INSTRUCTION = (
    "You are an expert Uyghur-to-English translator.\n"
    "Translate the following Uyghur sentence into fluent, accurate English.\n"
    "You may call the available tools to look up information about the words "
    "of the sentence — call a tool only when it helps you, as often as you "
    "need, and then give your final English translation as plain text.\n\n"
    "### Uyghur Source Sentence:\n{sentence}\n\n"
    "### English Translation:"
)

# Draft-first protocol (E4-DF): system message + protocol. The model first
# translates on its own (draft), then retrieves for its uncertainties, then
# gives a final translation. Separates the value of retrieval from framing
# effects: Δ(final − draft) per episode is retrieval value given a draft.
DF_SYSTEM = (
    "You are an expert Uyghur-to-English translator. Work in this order:\n"
    "1. In your first reply, translate the Uyghur sentence as well as you "
    "can on your own. Output only the draft translation, nothing else.\n"
    "2. Identify the specific words or constructions you were uncertain "
    "about, and call the available tools to check exactly those. Do not "
    "look up words you were already sure of.\n"
    "3. When you are done checking, reply with the improved English "
    "translation only, with no preamble, explanations, or word lists."
)

DF_USER = (
    "Translate the following Uyghur sentence into English following the "
    "working order you were given.\n\n"
    "### Uyghur Source Sentence:\n{sentence}\n\n"
    "### English Translation:"
)

DF_DRAFT_NUDGE = (
    "That was your draft translation. Now check the words you were "
    "uncertain about by calling the tools. If you were already fully "
    "certain about every word, reply with your final translation now."
)

FORCED_FINAL = ("Tool budget exhausted. Do not call any more tools. "
                "Provide the final English translation now.")


@dataclass
class Episode:
    sent_idx: int
    sentence: str
    reference: str
    messages: List[dict] = field(default_factory=list)
    n_calls: int = 0
    n_rounds: int = 0
    n_parse_failures: int = 0
    n_invalid_calls: int = 0
    n_unresolved: int = 0
    n_length_caps: int = 0
    think_chars: int = 0
    n_skipped_calls: int = 0
    forced_final: bool = False
    draft: Optional[str] = None      # draft-first protocol: first own attempt
    budget_limit: int = 0          # per-episode call budget (fixed or proportional)
    n_content_tokens: int = 0      # sentence-length basis for proportional budgets
    tool_mix: Dict[str, int] = field(default_factory=dict)
    addressing: Dict[str, int] = field(default_factory=dict)
    call_log: List[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    finish: str = "in_progress"  # final | budget_forced | round_cap | failed | context_overflow
    translation: Optional[str] = None


class AgentRunner:
    def __init__(self, llm, server, tools, model_id: str,
                 max_calls: int = 12, max_rounds: int = 6,
                 max_tokens: int = 512, context_limit: int = 16384,
                 thinking: bool = False,
                 sampling: Optional[Dict] = None,
                 budget_policy: str = "fixed",
                 budget_alpha: float = 1.0,
                 budget_floor: int = 4,
                 draft_first: bool = False,
                 allow_unbudgeted_final: bool = True):
        self.llm = llm
        self.server = server
        self.tools = tools
        self.model_id = model_id
        self.max_calls = max_calls
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.context_limit = context_limit
        self.thinking = thinking
        self.sampling = sampling or {"temperature": 0.0}
        self.think_marker = THINK_MARKER_BY_MODEL.get(model_id)
        self.budget_policy = budget_policy
        self.budget_alpha = budget_alpha
        self.budget_floor = budget_floor
        self.draft_first = draft_first
        self.allow_unbudgeted_final = allow_unbudgeted_final

        # per-family parser
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        parser_cls = ToolParserManager.get_tool_parser(PARSER_BY_MODEL[model_id])
        tokenizer = self.llm.get_tokenizer()
        self.parser = parser_cls(tokenizer)
        self._request = self._make_request()

        self.chat_kwargs = dict(CHAT_KWARGS_BY_MODEL.get(model_id, {}))
        if self.thinking:
            self.chat_kwargs["enable_thinking"] = True
        self.single_call = model_id in SINGLE_CALL_PER_TURN
        self.tool_names = {t["function"]["name"] for t in tools}

    def _make_request(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest)
        return ChatCompletionRequest(
            model="offline",
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {
                "name": "f", "description": "",
                "parameters": {"type": "object", "properties": {}}}}],
        )

    # ------------------------------------------------------------------
    def new_episode(self, sent_idx, sentence, reference,
                    n_content_tokens: int = 0) -> Episode:
        ep = Episode(sent_idx=sent_idx, sentence=sentence, reference=reference)
        if self.draft_first:
            ep.messages = [
                {"role": "system", "content": DF_SYSTEM},
                {"role": "user", "content": DF_USER.format(sentence=sentence)},
            ]
        else:
            ep.messages = [{"role": "user", "content":
                           INSTRUCTION.format(sentence=sentence)}]
        ep.n_content_tokens = n_content_tokens
        if self.budget_policy == "proportional":
            # one call per content word (alpha) — enough to gloss every word
            # once; over-fetchers still hit it and are measured
            ep.budget_limit = max(self.budget_floor,
                                  round(self.budget_alpha * n_content_tokens))
        else:
            ep.budget_limit = self.max_calls
        return ep

    def _generate(self, episodes: List[Episode]) -> List[str]:
        from vllm import SamplingParams
        sampling = SamplingParams(max_tokens=self.max_tokens, **self.sampling)
        convs = [ep.messages for ep in episodes]
        outs = self.llm.chat(
            convs, sampling, tools=self.tools,
            chat_template_kwargs=self.chat_kwargs or None,
            use_tqdm=False,
        )
        texts = []
        for ep, out in zip(episodes, outs):
            ep.tokens_in += len(out.prompt_token_ids) if out.prompt_token_ids else 0
            ep.tokens_out += len(out.outputs[0].token_ids)
            raw = out.outputs[0].text
            # thinking mode: Qwen3.5 emits reasoning that flows into the tool
            # call with no marker (verified live); the only reliable split is
            # the reasoning-end marker on final turns. For tool-call turns we
            # keep the raw stream (the parser skips reasoning text); for
            # marker-less final turns the whole text is the reasoning + answer
            # (rare) and is treated as content after stripping the reasoning.
            if self.think_marker and self.think_marker in raw:
                reasoning, content = raw.split(self.think_marker, 1)
                ep.think_chars += len(reasoning)
                texts.append(content)
            else:
                if out.outputs[0].finish_reason == "length":
                    # generation ran out of tokens before finishing
                    ep.n_length_caps += 1
                    texts.append("")
                else:
                    texts.append(raw)
        return texts

    def _parse_calls(self, text: str) -> Optional[List[dict]]:
        """Returns list of {id, name, arguments(dict)} or None if no calls.
        Raises on unparseable text."""
        info = self.parser.extract_tool_calls(text, self._request)
        if not info.tools_called:
            return None
        calls = []
        seen_ids = set()
        for i, tc in enumerate(info.tool_calls):
            args = tc.function.arguments or "{}"
            try:
                args_dict = json.loads(args) if isinstance(args, str) else dict(args)
            except json.JSONDecodeError:
                args_dict = {"_raw": str(args)}
            cid = tc.id or f"call_{i}"
            if cid in seen_ids:  # ensure tool_call_ids are unique
                cid = f"{cid}#{i}"
            seen_ids.add(cid)
            calls.append({"id": cid, "name": tc.function.name,
                          "arguments": args_dict})
        return calls

    def _append_assistant(self, ep: Episode, text: str, calls: List[dict]):
        msg = {"role": "assistant", "content": text or ""}
        if calls:
            # Qwen3.5's chat template requires arguments as a dict for
            # tool_calls in the message history (string arguments fail
            # template rendering with thinking enabled)
            msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": dict(c["arguments"] or {})
                              if isinstance(c["arguments"], dict)
                              else str(c["arguments"] or "{}")}}
                for c in calls]
        ep.messages.append(msg)

    def _execute_calls(self, ep: Episode, calls: List[dict],
                       budget_left: int) -> bool:
        """Append the assistant message (with the calls actually executed —
        beyond-budget calls are dropped BEFORE rendering so the message never
        references results that don't exist), execute them, append tool
        messages. Returns True if any call was skipped for budget."""
        kept = calls[:budget_left]
        skipped = calls[budget_left:]
        if self.single_call and len(kept) > 1:
            skipped = kept[1:] + skipped  # template limit: 1 call per message
            kept = kept[:1]
        ep.n_skipped_calls += len(skipped)

        self._append_assistant(ep, "", kept)

        for c in kept:
            ret_text, log = self.server.execute(
                ep.sent_idx, c["name"], c["arguments"] or {})
            log["call_id"] = c["id"]
            log["round"] = ep.n_rounds
            if log.get("resolution") in ("unresolved", "ambiguous",
                                          "token_id_invalid"):
                ep.n_unresolved += 1
            ep.tool_mix[c["name"]] = ep.tool_mix.get(c["name"], 0) + 1
            args = c["arguments"] or {}
            if "token_id" in args:
                ep.addressing["token_id"] = ep.addressing.get("token_id", 0) + 1
            elif args.get("word"):
                ep.addressing["word"] = ep.addressing.get("word", 0) + 1
            ep.call_log.append(log)
            ep.messages.append({"role": "tool", "tool_call_id": c["id"],
                                "content": ret_text})
            ep.n_calls += 1
        return bool(skipped)

    def _final_answer_text(self, raw: str) -> str:
        """Extract the final-answer text: in thinking mode, reasoning may
        precede the answer; use the reasoning-end marker if present."""
        if self.think_marker and self.think_marker in raw:
            return raw.split(self.think_marker, 1)[1].strip()
        return raw.strip()

    def _looks_like_tool_call(self, text: str) -> bool:
        """Heuristic: text claims to be a tool call but the parser found none
        (typically malformed JSON, e.g. Llama's \\uXXXX escapes)."""
        tool_names = [t["function"]["name"] for t in self.tools]
        if "{" not in text or '"' not in text:
            return False
        return any(n in text for n in tool_names)

    # ------------------------------------------------------------------
    def run(self, episodes: List[Episode]) -> List[Episode]:
        """Run all episodes to completion (round-synchronous)."""
        active = {ep.sent_idx: ep for ep in episodes}
        done: Dict[int, Episode] = {}

        while active:
            # guardrail: terminate episodes whose conversation would overflow
            # the context window (pathological garbage-echo loops) instead of
            # letting them crash the whole batch at render time
            for ep in list(active.values()):
                est_chars = sum(len(str(m.get("content") or "")) for m in ep.messages)
                est_chars += sum(len(str(m.get("tool_calls") or ""))
                                 for m in ep.messages)
                est_tokens = est_chars // 2  # conservative mixed-script estimate
                if est_tokens > self.context_limit - self.max_tokens - 512:
                    ep.translation = (ep.translation or "").strip()
                    ep.finish = "context_overflow"
                    done[ep.sent_idx] = ep
                    del active[ep.sent_idx]

            if not active:
                break
            batch = list(active.values())
            texts = self._generate(batch)
            still_active = []

            for ep, text in zip(batch, texts):
                ep.n_rounds += 1
                # In thinking mode, a final-answer turn (no tool calls) puts
                # reasoning before the marker and the answer after; a
                # tool-call turn has no marker (reasoning flows into the
                # call). Split lazily per case below.
                raw = text
                clean = _strip_think(text)

                if not clean:
                    # no content produced (thinking consumed the token budget,
                    # or an empty generation) — corrective turn; protocol
                    # data, not an answer
                    ep.messages.append({"role": "user", "content":
                                         "Your last message contained no "
                                         "usable output. Be concise: either "
                                         "call one tool or provide the final "
                                         "English translation."})
                    still_active.append(ep)
                    continue

                try:
                    calls = self._parse_calls(text)
                except Exception:  # parser failure = protocol data
                    ep.n_parse_failures += 1
                    self._append_assistant(ep, clean, [])
                    ep.messages.append({"role": "user", "content":
                                         "Your last message could not be "
                                         "parsed as a tool call or a "
                                         "translation. Either call a tool "
                                         "correctly or provide the final "
                                         "English translation."})
                    still_active.append(ep)
                    continue

                if calls:
                    # validate parsed calls against the real tool names: a
                    # parser can produce garbage function names (seen with
                    # mistral multi-call turns), and echoing them back into
                    # the conversation breaks template validation downstream
                    valid = [c for c in calls if c["name"] in self.tool_names]
                    for c in calls:
                        if c["name"] not in self.tool_names:
                            ep.n_invalid_calls += 1
                            ep.call_log.append({
                                "tool": str(c["name"])[:80],
                                "args": c["arguments"], "resolved": None,
                                "resolution": None,
                                "result": "invalid_name_dropped",
                                "round": ep.n_rounds})
                    calls = valid
                    if not calls:
                        ep.n_parse_failures += 1
                        self._append_assistant(ep, clean, [])
                        ep.messages.append({"role": "user", "content":
                                             "Your tool call named an unknown "
                                             "function. Available tools: "
                                             + ", ".join(sorted(self.tool_names))
                                             + ". Call one of them correctly "
                                               "or provide the final English "
                                               "translation."})
                        still_active.append(ep)
                        continue

                if not calls:
                    # final answer — unless the text is a failed tool-call
                    # attempt (e.g. malformed JSON with \\uXXXX escapes):
                    # route to the corrective handler instead of accepting
                    # garbage as a translation
                    if self._looks_like_tool_call(raw):
                        ep.n_parse_failures += 1
                        self._append_assistant(ep, self._final_answer_text(raw), [])
                        ep.messages.append({"role": "user", "content":
                                             "Your tool call could not be "
                                             "parsed. When providing a word "
                                             "argument, copy the word exactly "
                                             "as plain text from the "
                                             "sentence; do not escape "
                                             "non-ASCII characters. Either "
                                             "call a tool correctly or "
                                             "provide the final English "
                                             "translation."})
                        still_active.append(ep)
                        continue

                    final_text = self._final_answer_text(raw)
                    if (self.draft_first and ep.draft is None
                            and not ep.forced_final and ep.n_calls == 0):
                        # draft-first protocol: the first no-call turn with
                        # no executed calls yet IS the draft — store it, nudge
                        # the model to check its uncertainties (or stop).
                        ep.draft = final_text
                        self._append_assistant(ep, final_text, [])
                        ep.messages.append({"role": "user", "content":
                                            DF_DRAFT_NUDGE})
                        still_active.append(ep)
                        continue
                    ep.translation = final_text
                    ep.finish = ("budget_forced"
                                 if ep.n_calls >= ep.budget_limit else "final")
                    done[ep.sent_idx] = ep
                    continue

                if ep.forced_final:
                    # model ignored the "no more tools" instruction: the
                    # refusal to stop is protocol data; terminate the episode
                    ep.translation = self._final_answer_text(raw)
                    ep.finish = "budget_forced"
                    done[ep.sent_idx] = ep
                    continue

                # has calls: execute within budget
                budget_left = ep.budget_limit - ep.n_calls
                if (self.draft_first and ep.draft is None and clean.strip()
                        and budget_left > 0):
                    # model drafted and called tools in the same turn: the
                    # content part is the draft
                    ep.draft = clean.strip()
                if budget_left > 0:
                    skipped = self._execute_calls(ep, calls, budget_left)
                else:
                    skipped = True  # nothing to execute; no assistant message
                    # (appending an assistant message with zero executed
                    # calls would leave the conversation assistant-
                    # terminated, which some chat templates reject)

                if (skipped or ep.n_calls >= ep.budget_limit) and not ep.forced_final:
                    ep.forced_final = True
                    ep.messages.append({"role": "user", "content": FORCED_FINAL})

                if ep.n_rounds >= self.max_rounds and not ep.forced_final:
                    ep.forced_final = True
                    ep.messages.append({"role": "user", "content": FORCED_FINAL})

                if ep.n_rounds >= self.max_rounds + 2:
                    # hard stop: keep whatever text we have
                    ep.translation = self._final_answer_text(raw)
                    ep.finish = ("round_cap" if ep.n_calls else "failed")
                    done[ep.sent_idx] = ep
                    continue
                still_active.append(ep)

            active = {ep.sent_idx: ep for ep in still_active}

        return [done[i] for i in sorted(done)]


def episode_record(ep: Episode, model_id: str, condition: str) -> dict:
    return {
        "id": ep.sent_idx,
        "model": model_id,
        "condition": condition,
        "sentence": ep.sentence,
        "reference": ep.reference,
        "draft": ep.draft,
        "translation": ep.translation,
        "finish": ep.finish,
        "n_rounds": ep.n_rounds,
        "n_calls": ep.n_calls,
        "budget_limit": ep.budget_limit,
        "n_content_tokens": ep.n_content_tokens,
        "n_skipped_calls": ep.n_skipped_calls,
        "forced_final": ep.forced_final,
        "n_parse_failures": ep.n_parse_failures,
        "n_invalid_calls": ep.n_invalid_calls,
        "n_length_caps": ep.n_length_caps,
        "think_chars": ep.think_chars,
        "n_unresolved": ep.n_unresolved,
        "tool_mix": ep.tool_mix,
        "addressing": ep.addressing,
        "tokens_in": ep.tokens_in,
        "tokens_out": ep.tokens_out,
        "call_log": ep.call_log,
        "messages": ep.messages,
    }
