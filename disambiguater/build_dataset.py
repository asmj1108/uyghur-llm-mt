import sys
import subprocess
import asyncio
import difflib
import io
import json
import random
import re
import os
from openai import AsyncOpenAI
from streamparser import parse_file, reading_to_string
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
TEST_MODE        = False
OUTPUT_FILE      = "uyghur_disambiguation_dataset_v1.jsonl"
MAX_CONCURRENT   = 20          # asyncio handles this more efficiently than threads

OPENROUTER_API_KEY = 'sk-or-v1-'
MODEL_NAME = 'google/gemini-3.1-flash-lite'

PRICE_PER_1M_PROMPT = 0.25
PRICE_PER_1M_COMP = 1.5

DELIMITER = "SENTBOUNDARY"

MAP = {'post': 'Postposition', 'lpar': 'Left parenthesis', 'dem': 'Demonstrative', 'f': 'Feminine', 'al': 'Altres',
       'gpr_fut': 'Future verbal adjective', 'nom': 'Nominative', 'ger_past': 'Past gerund', 'ger_fut': 'Future gerund',
       'cop': 'Copula', 'np': 'Proper noun', 'acc': 'Accusative', 'v': 'Standard verb', 'loc': 'Locative',
       'cnjadv': 'Conjunctive adverb', 'adj': 'Adjective', 'gpr_impf': 'Imperfect verbal adjective',
       'imp': 'Imperative', 'gna_impf': 'Imperfect verbal adverb', 'def': 'Definite', 'num': 'Numeral',
       'cnjcoo': 'Co-ordinating conjunction', 'n': 'Noun', 'aor': 'Aorist', 'neg': 'Negative', 'vaux': 'Auxiliary verb',
       'postadv': 'Post-adverb', 'abbr': "Abbreviation (e.g. ''etc., Mr.'')", 'lquot': 'Left quote', 'abl': 'Ablative',
       'prn': 'Pronoun', 'mod': 'Modal word', 'gna_perf': 'Perfect verbal adverb', 'cog': 'Cognomen', 'dat': 'Dative',
       'qst': 'Interrogative/question particle', 'frm': 'Formal', 'ger_impf': 'Imperfect gerund',
       'gpr_past': 'Past verbal adjective', 'attr': 'Attributive', 'gen': 'Genitive', 'fut': 'Future',
       'ant': 'Anthroponym', 'cnjsub': 'Sub-ordinating conjunction', 'sent': 'Sentence-ending punctuation',
       'ger': 'Gerund', 'cm': 'Comma punctuation', 'adv': 'Adverb', 'rquot': 'Right quote', 'ifi': 'Past definite',
       'ins': 'Instrumental or Instructive', 'iv': 'Intransitive', 'rpar': 'Right parenthesis', 'ij': 'Interjection',
       'comp': 'Comparative', 'coll': 'Collective', 'itg': 'Interrogative', 'det': 'Determiner', 'guio': 'Hyphen',
       'ind': 'Indefinite', 'ger_perf': 'Perfect gerund', 'px2pl': 'Second person plural possessive',
       'pat': 'Patronymic', 'pass': 'Passive voice', 'tv': 'Transitive', 'prc_perf': 'Perfect participle',
       'pres': 'Present', 'px1sg': 'First person singular possessive', 'p2': 'Second person',
       'px3pl': 'Third person plural possessive', 'px3sg': 'Third person singular possessive', 'p3': 'Third person',
       'top': 'Toponym', 'sg': 'Singular', 'qnt': 'Quantifier', 'mf': 'Masculine or feminine', 'm': 'Masculine',
       'percent': 'Percentage', 'prc_impf': 'Imperfect participle', 'px1pl': 'First person plural possessive',
       'p1': 'First person', 'past': 'Past', 'pers': 'Personal', 'pl': 'Plural', 'org': 'Organisation',
       'px3sp': 'Third person possessive singular or plural', 'ref': 'Reflexive', 'ord': 'Ordinal',
       'px2sg': 'Second person singular possessive', 'advl': 'Adverbial', 'subst': 'Substantive'}

TURKISH_SPECIFIC_MAP = {
    'gpr_rsub': 'Relative substantival verbal adjective', # Often used for Agentive (-guqi/-ghuchi) "the one who does..."
    'gpr_rsub3': 'Relative substantival verbal adjective (variant 3)',
    'gpr_rsub4': 'Relative substantival verbal adjective (variant 4)',
    'ger_pabs': 'Past absolute gerund',
    'ger_inf': 'Infinitive gerund',     # Specifically for the infinitive verbal noun forms (-maq/-mäk)
    'dub': 'Dubitative',                # Expresses doubt/hearsay (e.g., -GhUdek)
    'abil': 'Abilitative',              # Expresses ability (e.g., -ala/-älä)
    'aor2': 'Secondary aorist',         # Allomorph/variant of the aorist tense
    'gna_neg': 'Negative verbal adverb',# Negative converb (e.g., -may / -mastin)
    'dek': 'Equative',                  # Expresses similarity "like" or "as" (e.g., -dek/-tek)
    'evid': 'Evidential',               # Expresses indirect observation / hearsay past (e.g., indirect past -iptu)
    'aorp': 'Aorist participle'         # Aorist verbal adjective
}

MAP.update(TURKISH_SPECIFIC_MAP)

# Single async client — no threading required
async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=20.0   # Hard socket timeout: prevents hanging connections
)

# ==========================================
# CHECKPOINT / RESUME
# ==========================================
def load_completed_history():
    """
    Returns a set of (sentence, target_word) pairs already saved.
    NOTE: If the same word appears twice in the same sentence, only one
    occurrence will be checkpointed. This is an acceptable edge-case tradeoff.
    """
    completed = set()
    if not os.path.exists(OUTPUT_FILE):
        return completed
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                completed.add((data["sentence"], data["target_word"]))
            except json.JSONDecodeError:
                pass  # Ignore corrupted tail lines from a previous crash
    print(f"🔄 Checkpoint: {len(completed)} entries already in dataset.")
    return completed

# ==========================================
# ALIGNMENT & PARSING (Unchanged)
# ==========================================
def parse_tags_to_readable(tags_list):
    if not tags_list: return "UNKNOWN"
    return "Features: " + ", ".join(MAP.get(t, t.upper()) for t in tags_list)

def format_reading(reading):
    parts = reading_to_string(reading).split("+")
    descriptions = []
    for part in parts:
        if "<" in part:
            lemma = part[:part.index("<")]
            tags  = part[part.index("<")+1 : part.rindex(">")].split("><")
            descriptions.append(f"Lemma: {lemma} | {parse_tags_to_readable(tags)}")
    return " + Combined with: ".join(descriptions)

def filter_and_flatten(tokens, split_char):
    orig_filtered, flat_list = [], []
    for tok in tokens:
        if not any(c.isalnum() for c in tok): continue
        orig_filtered.append(tok)
        for sw in tok.split(split_char):
            if sw.strip():
                flat_list.append({"orig_idx": len(orig_filtered) - 1, "subword": sw})
    return orig_filtered, flat_list

def get_components(pairs):
    components, cg, ca = [], set(), set()
    for g, a in pairs:
        if cg and g not in cg and a not in ca:
            components.append((sorted(cg), sorted(ca)))
            cg, ca = {g}, {a}
        else:
            cg.add(g); ca.add(a)
    if cg or ca:
        components.append((sorted(cg), sorted(ca)))
    return components

def read_conllu(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text, tokens = "", []
        for line in f:
            line = line.strip()
            if not line:
                if tokens: yield text, tokens
                text, tokens = "", []
                continue
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 8 and "-" not in parts[0] and "." not in parts[0]:
                    tokens.append({
                        "id": parts[0], "form": parts[1],
                        "upos": parts[3], "deprel": parts[7]
                    })

def get_apertium_lus_batched(texts, apertium_dir):
    full_input = f" {DELIMITER} ".join(texts)
    cmd = ["apertium", "-d", apertium_dir, "uig-tagger"]
    res = subprocess.run(cmd, input=full_input.encode("utf-8"),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0: return []
    stream = io.StringIO(res.stdout.decode("utf-8"))
    sentences_lus, current = [], []
    for _, lu in parse_file(stream, with_text=True):
        if lu is None: continue
        if lu.wordform == DELIMITER:
            sentences_lus.append(current); current = []
        else:
            current.append(lu)
    if current: sentences_lus.append(current)
    return sentences_lus

def apply_rule_1(readings, gold_tokens):
    """Copula filter: modifiers cannot be zero-copula predicates."""
    modifier_deprels = {"amod","nummod","det","case","nmod","obj","obl","advmod","mark"}
    deprels = {t["deprel"].split(":")[0] for t in gold_tokens}
    indices = list(range(len(readings)))
    if deprels & modifier_deprels:
        filtered = [i for i, r in enumerate(readings) if "<cop>" not in reading_to_string(r)]
        if filtered:
            return filtered
    return indices

# ==========================================
# PHASE 1: SYNC RULE FILTERING
# ==========================================
def run_phase_1(conllu_data, apertium_dir, completed_history):
    """
    Runs locally. No API calls.
    Returns:
        llm_task_queue : list of dicts with everything needed for Phase 2
        stats          : dict of counts
    """
    llm_task_queue = []
    stats = {"rule_resolved": 0, "already_done": 0}
    batch_size = 150

    # Rule-based entries are written immediately in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for batch_start in tqdm(range(0, len(conllu_data), batch_size), desc="📖 Parsing Trees"):
            batch = conllu_data[batch_start : batch_start + batch_size]
            texts = [t for t, _ in batch]
            apt_batch = get_apertium_lus_batched(texts, apertium_dir)

            for i, (text, gold_tokens) in enumerate(batch):
                if i >= len(apt_batch): break
                apt_lus = apt_batch[i]

                gf, gflat = filter_and_flatten([t["form"] for t in gold_tokens], "_")
                af, aflat = filter_and_flatten([lu.wordform for lu in apt_lus], " ")
                if not gflat or not aflat: continue

                sm = difflib.SequenceMatcher(
                    None,
                    [x["subword"] for x in gflat],
                    [x["subword"] for x in aflat]
                )
                pairs = [
                    pt
                    for tag, i1, i2, j1, j2 in sm.get_opcodes()
                    if tag == "equal"
                    for pt in zip(range(i1, i2), range(j1, j2))
                ]
                components = get_components([
                    (gflat[x]["orig_idx"], aflat[y]["orig_idx"]) for x, y in pairs
                ])

                for g_idxs, a_idxs in components:
                    g_toks = [gold_tokens[idx] for idx in g_idxs]
                    for a_idx in a_idxs:
                        lu = apt_lus[a_idx]
                        if len(lu.readings) <= 1:
                            continue

                        # Skip if already processed in a previous run
                        if (text, lu.wordform) in completed_history:
                            stats["already_done"] += 1
                            continue

                        candidates       = [format_reading(r) for r in lu.readings]
                        surviving        = apply_rule_1(lu.readings, g_toks)
                        base_entry       = {
                            "sentence":    text,
                            "target_word": lu.wordform,
                            "candidates":  candidates,
                        }

                        if len(surviving) == 1:
                            # Rule fully resolved it — write immediately
                            entry = {**base_entry,
                                     "label_id":      surviving[0],
                                     "source":        "rule_based",
                                     "llm_reasoning": "Rule 1: Copula eliminated by modifier DEPREL."}
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            f.flush()   # ← BUG FIX: persist immediately
                            stats["rule_resolved"] += 1
                        else:
                            # Needs LLM — queue it
                            llm_task_queue.append({
                                "entry":            base_entry,
                                "target_word":      lu.wordform,
                                "sentence":         text,
                                "subset_texts":     [candidates[i] for i in surviving],
                                "surviving_mapping": surviving,
                            })

    return llm_task_queue, stats

# ==========================================
# PHASE 2: ASYNC LLM WORKER
# ==========================================
async def call_llm(task, semaphore):
    """
    One async coroutine per word.
    asyncio.Semaphore limits concurrency to MAX_CONCURRENT without threads.
    Returns (llm_result_dict | None, task, prompt_tokens, completion_tokens)
    """
    async with semaphore:
        candidates_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(task["subset_texts"])
        )
        prompt = f"""You are an expert computational linguist specializing in Uyghur morphology.
Review the sentence, target word, and candidate grammatical analyses.
Sentence: {task['sentence']}
Target Word: {task['target_word']}

Candidate Analyses:
{candidates_text}

Analyze the grammatical role of the target word, then select the ONE correct candidate integer.
Respond in valid JSON containing exactly two keys: "reasoning" (string) and "correct_id" (integer)."""


        for attempt in range(4):
            try:
                response = await async_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    reasoning_effort="high",
                    extra_body={
                        "reasoning": {
                            "effort": "high"
                        }
                    },
                )
                content = response.choices[0].message.content
                match   = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return (
                        json.loads(match.group()),
                        task,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )
                break  # Parseable response but no JSON — don't retry

            except Exception as e:
                err = str(e)
                if "429" in err or "timeout" in err.lower():
                    # Exponential backoff with jitter — fully async, doesn't block event loop
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
                else:
                    break

        return None, task, 0, 0


async def run_phase_2_async(llm_task_queue):
    """
    Runs all LLM tasks concurrently using asyncio.
    No threads, no GIL, no lock contention.
    """
    semaphore   = asyncio.Semaphore(MAX_CONCURRENT)
    coroutines  = [call_llm(task, semaphore) for task in llm_task_queue]

    p_tok_total = 0
    c_tok_total = 0
    resolved    = 0
    errors      = 0

    pbar = tqdm(total=len(coroutines), desc="🤖 LLM API Calls")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # asyncio.as_completed yields each coroutine the moment it finishes
        # No internal lock contention — scales to 10,000+ tasks easily
        for coro in asyncio.as_completed(coroutines):
            llm_res, task_info, p_tok, c_tok = await coro
            p_tok_total += p_tok
            c_tok_total += c_tok

            if llm_res and "correct_id" in llm_res:
                relative_id = llm_res["correct_id"] - 1
                if 0 <= relative_id < len(task_info["subset_texts"]):
                    true_id = task_info["surviving_mapping"][relative_id]
                    entry   = {
                        **task_info["entry"],
                        "label_id":      true_id,
                        "source":        "llm_based",
                        "llm_reasoning": llm_res.get("reasoning", ""),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()   # ← BUG FIX: each result persists instantly
                    resolved += 1
                else:
                    errors += 1
            else:
                errors += 1

            pbar.update(1)

    pbar.close()
    return p_tok_total, c_tok_total, resolved, errors

# ==========================================
# MAIN
# ==========================================
def build_dataset(conllu_path, apertium_dir):
    print("Loading CoNLL-U data...")
    conllu_data       = list(read_conllu(conllu_path))
    completed_history = load_completed_history()

    # ---- Phase 1 (Fast, local, deterministic) ----
    print("\n--- PHASE 1: Rule Filtering ---")
    llm_queue, p1_stats = run_phase_1(conllu_data, apertium_dir, completed_history)

    print(f"\n✅ Phase 1 Complete:")
    print(f"   Skipped (already done) : {p1_stats['already_done']}")
    print(f"   Rule-resolved          : {p1_stats['rule_resolved']}")
    print(f"   Queued for LLM         : {len(llm_queue)}")

    # ---- Phase 2 (Async LLM) ----
    p_tok = c_tok = llm_resolved = api_errors = 0

    if llm_queue:
        print(f"\n--- PHASE 2: Async LLM Processing ({len(llm_queue)} words) ---")
        p_tok, c_tok, llm_resolved, api_errors = asyncio.run(
            run_phase_2_async(llm_queue)
        )
    else:
        print("\n✅ No LLM tasks remaining. Dataset is up to date!")

    # ---- Final Report ----
    print("\n" + "="*50)
    print("  🎯  DATASET GENERATION COMPLETE  🎯")
    print("="*50)
    print(f"  Rule-Based Labels  : {p1_stats['rule_resolved']}")
    print(f"  LLM-Based Labels   : {llm_resolved}")
    print(f"  API Errors         : {api_errors}")

    if llm_resolved > 0:
        cost = (p_tok / 1_000_000) * PRICE_PER_1M_PROMPT \
               + (c_tok / 1_000_000) * PRICE_PER_1M_COMP
        print(f"\n  💸 API Cost This Run : ${cost:.5f} USD")
        print(f"  Prompt Tokens       : {p_tok:,}")
        print(f"  Completion Tokens   : {c_tok:,}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_dataset.py <conllu_path> <apertium_dir>")
        sys.exit(1)
    build_dataset(sys.argv[1], sys.argv[2])