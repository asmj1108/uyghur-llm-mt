import sys
import os
import io
import re
import json
import random
import difflib
import asyncio
import subprocess

from openai import AsyncOpenAI
from streamparser import parse_file, reading_to_string
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
TEST_MODE = False
TEST_LIMIT = 1  # sentences when TEST_MODE
OUTPUT_FILE = "uyghur_disambiguation_dataset.jsonl"
LABEL_VOCAB_FILE = "label_vocab.json"
MAX_CONCURRENT = 20

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = "google/gemini-3.1-flash-lite"

PRICE_PER_1M_PROMPT = 0.25
PRICE_PER_1M_COMP = 1.5

DELIMITER = "SENTBOUNDARY"

# ==========================================
# TAG -> GLOSS MAP  (Apertium symbols -> UD-style human-readable glosses)
# ==========================================
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
    'gpr_rsub': 'Relative substantival verbal adjective',
    'ger_pabs': 'Past absolute gerund',
    'ger_inf': 'Infinitive gerund',
    'dub': 'Dubitative',
    'abil': 'Abilitative',
    'gna_neg': 'Negative verbal adverb',
    'dek': 'Equative',
    'evid': 'Evidential',
    'aorp': 'Aorist participle',
    'gna_cond': 'Conditional verbal adverb',
    'iver': 'Continuative aspect',
    'prog': 'Progressive',
    'ter': 'Terminative converb',
    'che': 'Limitative',
    'td': 'Unspecified transitivity'
}

MAP.update(TURKISH_SPECIFIC_MAP)

GARBAGE_TAGS = {
    'err_orth',  # non-standard spelling accepted by the analyzer
}


def gloss_tag(tag):
    """Apertium tag -> human gloss, or None if it should be dropped."""
    t = tag.lower()
    if t in GARBAGE_TAGS or tag in GARBAGE_TAGS:
        return None
    if t in MAP:
        return MAP[t]
    base = re.sub(r'\d+$', '', t)  # strip trailing variant number
    if base != t and base in MAP:
        return MAP[base]
    return t.upper()  # surface raw so it stays visible


# ==========================================
# MUDT RULE CONSTANTS
# ==========================================
MODIFIER_DEPRELS = {'amod', 'nummod', 'det', 'case', 'nmod', 'obj', 'obl', 'advmod', 'mark', 'nsubj', 'csubj', 'iobj'}

# ==========================================
# PROMPT (finalized selection principle, C.6)
# ==========================================
SELECTION_PRINCIPLE = (
    "Choose the single candidate whose features ALL correctly describe the target "
    "word in this context. Reject any candidate that contains even one incorrect "
    "feature, even if its other features are correct. A candidate with fewer "
    "features can be the correct answer — do not prefer a candidate simply because "
    "it carries more features. Crucially, the selected candidate must account for "
    "the fully inflected target word, including all of its suffixes, rather than "
    "just describing its base lemma. Pay strict attention to the word's syntactic "
    "role in the sentence to resolve ambiguities in identical suffixes. If a candidate "
    "contains multiple components separated by '+ Combined with:', ensure the entire "
    "combined structure makes sense in context."
)

# Deterministic rationales for non-LLM sources (so all rows carry a rationale
# usable for decoder rationale-supervision).
DEDUP_REASONING = ("All candidate analyses are linguistically equivalent after "
                   "feature-set normalization; a single canonical reading remains.")


def rule_reasoning(rule_tag):
    if rule_tag == "NEG":
        return ("The target word bears a modifier/argument relation, so it cannot host "
                "the null third-person copula; all copula readings are eliminated.")
    if rule_tag == "POS_strong":
        return ("The target word is the head of a cop:zero arc, so it is the nominal "
                "predicate hosting the null third-person copula; only copula readings are kept.")
    if rule_tag == "POS_weak":
        return ("The target word is the clause root and the analyzer offers a null "
                "third-person copula reading (emitted only for nominal predicates), so it "
                "is the predicate; only copula readings are kept.")
    return ""


# ==========================================
# ASYNC CLIENT
# ==========================================
async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=30.0,
)


# ==========================================
# CHECKPOINT / RESUME
# ==========================================
def load_completed_history():
    completed = set()
    if not os.path.exists(OUTPUT_FILE):
        return completed
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                completed.add((data["sentence"], data["target_word"],
                               data.get("target_occurrence", 0)))
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"🔄 Checkpoint: {len(completed)} entries already in dataset.")
    return completed


# ==========================================
# READING -> (natural string, feature-label list, has_cop)   [C.1]
# ==========================================
def parse_reading(reading):
    """
    Parse an Apertium reading into:
      natural   : human-readable gloss string (lemma included)   -> decoder models
      feats     : canonical feature-label list                   -> encoder multi-label
                  * MAIN unit  : lemma OMITTED, each gloss is a separate label
                  * ENCLITIC   : kept as ONE atomic label that RETAINS its lemma
                                 (e.g. copula 'Lemma: ئى | Features: Copula, ...')
      has_cop   : True if the reading contains a <cop> tag
    """
    raw = reading_to_string(reading)
    parts = raw.split("+")
    natural_parts, feats = [], []
    main_ok = False

    for pidx, part in enumerate(parts):
        if "<" not in part:
            continue
        lemma = part[:part.index("<")]
        tags = part[part.index("<") + 1: part.rindex(">")].split("><")
        gloss = [g for g in (gloss_tag(t) for t in tags) if g is not None]

        if pidx == 0:
            # main unit must have a real lemma AND at least one valid feature
            if not lemma.strip() or not gloss:
                return None  # ← drop the whole reading
            main_ok = True
            natural_parts.append(f"Lemma: {lemma} | Features: {', '.join(gloss)}")
            feats.extend(gloss)
        else:
            if not gloss:
                continue
            atomic = f"Lemma: {lemma} | Features: {', '.join(gloss)}"
            natural_parts.append(atomic)
            feats.append(atomic)

    if not main_ok:
        return None
    natural = " + Combined with: ".join(natural_parts)
    return natural, feats, ("<cop>" in raw)


def dedup_candidates(readings):
    """
    Collapse linguistically identical readings by normalized feature set (C.6).
    Order preserved by first occurrence. Returns list of candidate dicts.
    """
    unique, seen = [], set()
    for r in readings:
        parsed = parse_reading(r)
        if parsed is None:  # ← skip dropped readings
            continue
        natural, feats, has_cop = parsed
        key = frozenset(feats)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"natural": natural, "feats": feats, "has_cop": has_cop})
    return unique


# ==========================================
# ALIGNMENT HELPERS
# ==========================================
def filter_and_flatten(tokens, split_char):
    orig_filtered, flat_list = [], []
    for true_idx, tok in enumerate(tokens):  # ← enumerate the FULL list
        if not any(c.isalnum() for c in tok):
            continue
        orig_filtered.append(tok)
        for sw in tok.split(split_char):
            if sw.strip():
                flat_list.append({"orig_idx": true_idx, "subword": sw})  # ← true index
    return orig_filtered, flat_list


def get_components(pairs):
    components, cg, ca = [], set(), set()
    for g, a in pairs:
        if cg and g not in cg and a not in ca:
            components.append((sorted(cg), sorted(ca)))
            cg, ca = {g}, {a}
        else:
            cg.add(g)
            ca.add(a)
    if cg or ca:
        components.append((sorted(cg), sorted(ca)))
    return components


# ==========================================
# CoNLL-U PARSING (+ predicate marking, C.4)
# ==========================================
def mark_predicates(tokens):
    """
    STRONG predicate signal (UPOS-free): token is HEAD of a cop:zero arc.
    The WEAK signal (nominal root) is deferred to apply_mudt_rules, where the
    presence of an Apertium copula candidate serves as the nominality proxy.
    Result: the whole filter needs only HEAD/DEPREL + Apertium readings —
    never UPOS, which is unavailable at inference (DiaParser gives HEAD/DEPREL only).
    """
    cop_zero_heads = {t['head'] for t in tokens if t['deprel'] == 'cop:zero'}
    for t in tokens:
        t['pred_strength'] = 'strong' if t['id'] in cop_zero_heads else None
    return tokens


def read_conllu(file_path):
    out = []
    with open(file_path, "r", encoding="utf-8") as f:
        text, tokens = "", []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    out.append((text, mark_predicates(tokens)))
                text, tokens = "", []
                continue
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            elif line.startswith("#"):
                continue
            else:
                parts = line.split("\t")
                if len(parts) >= 8 and "-" not in parts[0] and "." not in parts[0]:
                    tokens.append({
                        "id": parts[0], "form": parts[1], "lemma": parts[2],
                        "upos": parts[3], "head": parts[6], "deprel": parts[7],
                    })
        if tokens:
            out.append((text, mark_predicates(tokens)))
    return out


def get_apertium_lus_batched(texts, apertium_dir):
    full_input = f" {DELIMITER} ".join(texts)
    cmd = ["apertium", "-d", apertium_dir, "uig-tagger"]
    res = subprocess.run(cmd, input=full_input.encode("utf-8"),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("⚠️  Apertium failed on batch.")
        return []
    stream = io.StringIO(res.stdout.decode("utf-8"))
    sentences_lus, current = [], []
    for _, lu in parse_file(stream, with_text=True):
        if lu is None:
            continue
        if lu.wordform == DELIMITER:
            sentences_lus.append(current)
            current = []
        else:
            current.append(lu)
    if current:
        sentences_lus.append(current)
    return sentences_lus


def get_apertium_lus_safe(texts, apertium_dir):
    """Tries batched processing; falls back to 1-by-1 if synchronization is lost."""
    apt_batch = get_apertium_lus_batched(texts, apertium_dir)

    if len(apt_batch) == len(texts):
        return apt_batch

    print(f"⚠️ Apertium batch mismatch (got {len(apt_batch)}, expected {len(texts)}). Fixing 1-by-1...")
    safe_batch = []
    for text in texts:
        res = get_apertium_lus_batched([text], apertium_dir)
        if res:
            safe_batch.append(res[0])
        else:
            safe_batch.append([])  # empty list protects the index synchronization
    return safe_batch


# ==========================================
# MUDT ZERO-COPULA RULES (C.4)  — operate on deduped candidate dicts
# ==========================================
def apply_mudt_rules(candidates, g_tokens):
    """
    Returns (surviving_indices, rule_tag) where rule_tag ∈
        {'NEG', 'POS_strong', 'POS_weak', None}.
    Inputs are HEAD/DEPREL (from the tree) + Apertium readings — no UPOS.
    """
    n = len(candidates)
    indices = list(range(n))
    if n <= 1:
        return indices, None

    deprels = {t['deprel'].split(':')[0] for t in g_tokens}
    has_cop_cand = any(c['has_cop'] for c in candidates)
    is_strong = any(t.get('pred_strength') == 'strong' for t in g_tokens)
    is_root = 'root' in deprels

    # ---- Rule 1b POSITIVE (predicate -> keep only copula) ----
    # Nominality is GUARANTEED by has_cop_cand: Apertium emits the null
    # 3rd-person copula only on bare nominal/adjectival/numeral predicates,
    # so an explicit nominal-UPOS test is redundant here.
    if (is_strong or is_root) and has_cop_cand:
        filtered = [i for i, c in enumerate(candidates) if c['has_cop']]
        if filtered and len(filtered) < n:
            strength = 'strong' if is_strong else 'weak'
            return filtered, f"POS_{strength}"

    # ---- Rule 1a NEGATIVE (modifier/arg -> drop copula) ----
    elif (deprels & MODIFIER_DEPRELS) and has_cop_cand:
        filtered = [i for i, c in enumerate(candidates) if not c['has_cop']]
        if filtered and len(filtered) < n:
            return filtered, "NEG"

    return indices, None


# ==========================================
# ENTRY CONSTRUCTION (C.7 schema)
# ==========================================
def make_entry(base, label_id, reasoning, source):
    """
    Creates the final dataset row.

    CONTRACT FOR DECODER FINETUNING SCRIPT:
    The `ANSWER: N` string targets the FULL, globally deduped `base["candidates"]` list.
    Even if the LLM originally chose from a reduced rule-filtered subset, this function
    maps the answer back to the global index.
    THEREFORE: Your Qwen/ByT5 finetuning script MUST show the model the full
    `candidates` list (1-indexed) in its prompt to align perfectly with the target text.
    """
    entry = dict(base)
    entry["label_id"] = label_id
    entry["reasoning"] = reasoning
    entry["target_text"] = f"REASON: {reasoning} | ANSWER: {label_id + 1}"
    entry["source"] = source
    entry["label_feats"] = base["candidate_feats"][label_id]
    return entry


# ==========================================
# PHASE 1 : LOCAL — dedup + rules (no API)
# ==========================================
def run_phase_1(conllu_data, apertium_dir, completed_history):
    llm_task_queue = []
    stats = {k: 0 for k in [
        "ambiguous", "already_done", "align_failed",
        "dedup_resolved", "rule_resolved", "queued",
        "neg", "pos_strong", "pos_weak"]}
    batch_size = 150

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for bs in tqdm(range(0, len(conllu_data), batch_size), desc="📖 Parsing trees"):
            batch = conllu_data[bs: bs + batch_size]
            texts = [t for t, _ in batch]
            apt_batch = get_apertium_lus_safe(texts, apertium_dir)

            for i, (text, gold_tokens) in enumerate(batch):
                apt_lus = apt_batch[i]
                if not apt_lus:
                    continue

                # occurrence ordinal of each apertium token's wordform (left-to-right)
                wf_counts, occ_of_aidx = {}, {}
                for idx, lu in enumerate(apt_lus):
                    wf = lu.wordform
                    occ_of_aidx[idx] = wf_counts.get(wf, 0)
                    wf_counts[wf] = wf_counts.get(wf, 0) + 1

                gf, gflat = filter_and_flatten([t["form"] for t in gold_tokens], "_")
                af, aflat = filter_and_flatten([lu.wordform for lu in apt_lus], " ")
                if not gflat or not aflat:
                    continue

                sm = difflib.SequenceMatcher(
                    None, [x["subword"] for x in gflat], [x["subword"] for x in aflat])
                pairs, aligned = [], True
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        pairs.extend(zip(range(i1, i2), range(j1, j2)))
                    else:
                        aligned = False
                        break
                if not aligned:
                    stats["align_failed"] += 1
                    continue

                components = get_components(
                    [(gflat[x]["orig_idx"], aflat[y]["orig_idx"]) for x, y in pairs])

                for g_idxs, a_idxs in components:
                    g_toks = [gold_tokens[idx] for idx in g_idxs]
                    for a_idx in a_idxs:
                        lu = apt_lus[a_idx]
                        if len(lu.readings) <= 1:
                            continue
                        occ = occ_of_aidx[a_idx]
                        if (text, lu.wordform, occ) in completed_history:
                            stats["already_done"] += 1
                            continue

                        stats["ambiguous"] += 1

                        unique = dedup_candidates(lu.readings)
                        candidates_nat = [c["natural"] for c in unique]
                        candidate_feats = [c["feats"] for c in unique]
                        base_entry = {
                            "sentence": text,
                            "target_word": lu.wordform,
                            "target_occurrence": occ,
                            "candidates": candidates_nat,
                            "candidate_feats": candidate_feats,
                        }

                        if len(unique) == 1:
                            entry = make_entry(base_entry, 0, DEDUP_REASONING, "dedup")
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            f.flush()
                            stats["dedup_resolved"] += 1
                            continue

                        # ---- 2. MUDT RULES ----
                        surviving, rule_tag = apply_mudt_rules(unique, g_toks)
                        if len(surviving) == 1:
                            entry = make_entry(base_entry, surviving[0],
                                               rule_reasoning(rule_tag), "rule_based")
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            f.flush()
                            stats["rule_resolved"] += 1
                            if rule_tag == "NEG":
                                stats["neg"] += 1
                            elif rule_tag == "POS_strong":
                                stats["pos_strong"] += 1
                            elif rule_tag == "POS_weak":
                                stats["pos_weak"] += 1
                            continue

                        # ---- 3. LLM RESIDUAL (full or rule-reduced subset) ----
                        llm_task_queue.append({
                            "base_entry": base_entry,
                            "sentence": text,
                            "target_word": lu.wordform,
                            "subset_texts": [candidates_nat[k] for k in surviving],
                            "surviving_mapping": surviving,
                        })
                        stats["queued"] += 1

    return llm_task_queue, stats


# ==========================================
# PHASE 2 : ASYNC LLM
# ==========================================
def mark_sentence_for_llm(sentence, target_word, occurrence=0):
    tw = target_word.strip()
    pattern = re.compile(rf"(?<!\S){re.escape(tw)}(?!\S)")
    matches = list(pattern.finditer(sentence))
    if matches:
        m = matches[occurrence] if occurrence < len(matches) else matches[0]
        return sentence[:m.start()] + f"<t> {tw} </t>" + sentence[m.end():]
    if tw in sentence:
        return sentence.replace(tw, f"<t> {tw} </t>", 1)
    return f"{sentence} «{tw}»"


async def call_llm(task, semaphore):
    async with semaphore:
        candidates_text = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(task["subset_texts"]))
        marked = mark_sentence_for_llm(task["sentence"], task["target_word"], task["base_entry"]["target_occurrence"])
        prompt = f"""You are an expert computational linguist specializing in Uyghur morphosyntax, Uyghur agglutinative morphology, and contextual disambiguation.

Sentence: {marked}
Target word (marked with <t> ... </t>): {task['target_word']}

Candidate analyses:
{candidates_text}

{SELECTION_PRINCIPLE}

Respond in valid JSON with EXACTLY two keys:
- "reasoning": ONE sentence explaining why the chosen candidate's features are ALL correct.
- "correct_id": the integer id (1-indexed) of the correct candidate. You MUST select an ID from the provided list."""

        for attempt in range(4):
            try:
                response = await async_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    reasoning_effort="medium",
                )
                content = response.choices[0].message.content
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return (json.loads(match.group()), task,
                            response.usage.prompt_tokens, response.usage.completion_tokens)
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "timeout" in err.lower():
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
                else:
                    break
        return None, task, 0, 0


async def run_phase_2_async(llm_task_queue):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    coroutines = [call_llm(t, semaphore) for t in llm_task_queue]

    p_tok = c_tok = resolved = errors = 0
    pbar = tqdm(total=len(coroutines), desc="🤖 LLM API calls")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(coroutines):
            llm_res, task, pt, ct = await coro
            p_tok += pt
            c_tok += ct

            if llm_res and "correct_id" in llm_res:
                rel = llm_res["correct_id"] - 1
                if 0 <= rel < len(task["subset_texts"]):
                    label_id = task["surviving_mapping"][rel]
                    reasoning = llm_res.get("reasoning", "")
                    entry = make_entry(task["base_entry"], label_id, reasoning, "llm_based")
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    resolved += 1
                else:
                    errors += 1
            else:
                errors += 1
            pbar.update(1)

    pbar.close()
    return p_tok, c_tok, resolved, errors


# ==========================================
# GLOBAL MULTI-LABEL VOCABULARY (for XLM-R)
# ==========================================
def write_label_vocab(output_file, vocab_file):
    vocab = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for feats in d.get("candidate_feats", []):
                vocab.update(feats)
    vocab = sorted(vocab)
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({"num_labels": len(vocab),
                   "label2id": {lab: i for i, lab in enumerate(vocab)},
                   "labels": vocab},
                  f, ensure_ascii=False, indent=2)
    print(f"📑 Wrote {len(vocab)} multi-label classes to {vocab_file}")


# ==========================================
# MAIN
# ==========================================
def build_dataset(conllu_path, apertium_dir):
    print("Loading CoNLL-U data...")
    conllu_data = read_conllu(conllu_path)
    if TEST_MODE:
        conllu_data = conllu_data[:TEST_LIMIT]
        print(f"🧪 TEST_MODE: limited to {len(conllu_data)} sentences.")

    completed_history = load_completed_history()

    # ---- Phase 1 ----
    print("\n--- PHASE 1: Dedup + MUDT Rules (local) ---")
    llm_queue, s = run_phase_1(conllu_data, apertium_dir, completed_history)

    print("\n✅ Phase 1 complete:")
    print(f"   Ambiguous words        : {s['ambiguous']}")
    print(f"   Skipped (already done) : {s['already_done']}")
    print(f"   Alignment failures     : {s['align_failed']}")
    print(f"   Dedup-resolved         : {s['dedup_resolved']}")
    print(f"   Rule-resolved          : {s['rule_resolved']} "
          f"(NEG={s['neg']}, POS_strong={s['pos_strong']}, POS_weak={s['pos_weak']})")
    print(f"   Queued for LLM         : {s['queued']}")

    # ---- Phase 2 ----
    p_tok = c_tok = llm_resolved = api_errors = 0
    if llm_queue:
        print(f"\n--- PHASE 2: Async LLM ({len(llm_queue)} words) ---")
        p_tok, c_tok, llm_resolved, api_errors = asyncio.run(run_phase_2_async(llm_queue))
    else:
        print("\n✅ No LLM tasks remaining. Dataset is up to date!")

    # ---- Global label space ----
    write_label_vocab(OUTPUT_FILE, LABEL_VOCAB_FILE)

    # ---- Report ----
    print("\n" + "=" * 50)
    print("  🎯  DATASET GENERATION COMPLETE  🎯")
    print("=" * 50)
    print(f"  Dedup-Based Labels : {s['dedup_resolved']}")
    print(f"  Rule-Based Labels  : {s['rule_resolved']}")
    print(f"  LLM-Based Labels   : {llm_resolved}")
    print(f"  API Errors         : {api_errors}")
    if llm_resolved > 0:
        cost = (p_tok / 1_000_000) * PRICE_PER_1M_PROMPT + (c_tok / 1_000_000) * PRICE_PER_1M_COMP
        print(f"\n  💸 API cost this run : ${cost:.5f} USD")
        print(f"  Prompt tokens       : {p_tok:,}")
        print(f"  Completion tokens   : {c_tok:,}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_dataset.py <conllu_path> <apertium_dir>")
        sys.exit(1)
    build_dataset(sys.argv[1], sys.argv[2])
