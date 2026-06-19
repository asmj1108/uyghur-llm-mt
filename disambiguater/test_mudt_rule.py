import sys
import subprocess
import difflib
import io
from streamparser import parse_file, reading_to_string

DELIMITER = "SENTBOUNDARY"

# ==========================================
# ALIGNMENT HELPERS (unchanged)
# ==========================================
def filter_and_flatten(tokens, split_char):
    orig_filtered = []
    flat_list = []
    for tok in tokens:
        if not any(c.isalnum() for c in tok):
            continue
        orig_filtered.append(tok)
        current_idx = len(orig_filtered) - 1
        for subword in tok.split(split_char):
            if subword.strip():
                flat_list.append({'orig_idx': current_idx, 'subword': subword})
    return orig_filtered, flat_list

def get_components(pairs):
    components = []
    current_g, current_a = set(), set()
    for g, a in pairs:
        if current_g and g not in current_g and a not in current_a:
            components.append((sorted(current_g), sorted(current_a)))
            current_g, current_a = {g}, {a}
        else:
            current_g.add(g)
            current_a.add(a)
    if current_g or current_a:
        components.append((sorted(current_g), sorted(current_a)))
    return components

# ==========================================
# DATA PARSING (now captures HEAD + computes predicates)
# ==========================================
NOMINAL_UPOS = {'NOUN', 'ADJ', 'NUM', 'PROPN'}

def mark_predicates(tokens):
    """
    Flags tokens that function as nominal predicates:
      - STRONG signal: token is the HEAD of a `cop:zero` arc.
      - WEAK signal:   token is `root` with nominal UPOS (subject likely pro-dropped).
    """
    cop_zero_heads = {t['head'] for t in tokens if t['deprel'] == 'cop:zero'}
    for t in tokens:
        strong = t['id'] in cop_zero_heads
        weak = (t['deprel'] == 'root' and t['upos'] in NOMINAL_UPOS)
        t['is_predicate'] = strong or weak
        t['pred_strength'] = 'strong' if strong else ('weak' if weak else None)
    return tokens

def read_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = ""
        tokens = []
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if tokens:
                    yield text, mark_predicates(tokens)
                text, tokens = "", []
                continue
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            elif line.startswith("#"):
                continue
            else:
                parts = line.split("\t")
                if len(parts) >= 8 and '-' not in parts[0] and '.' not in parts[0]:
                    tokens.append({
                        'id': parts[0], 'form': parts[1], 'lemma': parts[2],
                        'upos': parts[3], 'head': parts[6], 'deprel': parts[7]
                    })
        if tokens:
            yield text, mark_predicates(tokens)

def get_apertium_lus_batched(texts, apertium_dir):
    print(f"--> Sending batch of {len(texts)} sentences to Apertium...")
    full_input = f" {DELIMITER} ".join(texts)
    cmd = ["apertium", "-d", apertium_dir, "uig-tagger"]
    result = subprocess.run(cmd, input=full_input.encode('utf-8'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Apertium failed on batch.")
        return []

    stream = io.StringIO(result.stdout.decode('utf-8'))
    all_lus = [(blank, lu) for blank, lu in parse_file(stream, with_text=True) if lu is not None]

    sentences_lus, current = [], []
    for _, lu in all_lus:
        if lu.wordform == DELIMITER:
            sentences_lus.append(current)
            current = []
        else:
            current.append(lu)
    if current:
        sentences_lus.append(current)
    return sentences_lus

# ==========================================
# MUDT ZERO-COPULA RULES
# ==========================================
MODIFIER_DEPRELS = {'amod', 'nummod', 'det', 'case', 'nmod',
                    'obj', 'obl', 'advmod', 'mark'}

def has_cop(reading):
    return "<cop>" in reading_to_string(reading)

def apply_mudt_rules(g_tokens, lu):
    """
    Returns: (initial_count, final_count, surviving_readings, rule_tag)
    rule_tag in {'NEG', 'POS_strong', 'POS_weak', None}
    """
    readings = list(lu.readings)
    init = len(readings)
    if init <= 1:
        return init, init, readings, None

    is_pred = any(t['is_predicate'] for t in g_tokens)
    deprels = {t['deprel'].split(':')[0] for t in g_tokens}
    has_copula_candidate = any(has_cop(r) for r in readings)

    # ---- Rule 1b: POSITIVE (predicate -> keep only copula) ----
    # Only fires when a copula reading actually exists among candidates.
    if is_pred and has_copula_candidate:
        filtered = [r for r in readings if has_cop(r)]
        if filtered and len(filtered) < len(readings):
            strength = 'strong' if any(t.get('pred_strength') == 'strong' for t in g_tokens) else 'weak'
            return init, len(filtered), filtered, f"POS_{strength}"

    # ---- Rule 1a: NEGATIVE (modifier -> drop copula) ----
    # Exclude predicates explicitly so the two rules never conflict.
    elif (not is_pred) and deprels.intersection(MODIFIER_DEPRELS):
        filtered = [r for r in readings if not has_cop(r)]
        if filtered and len(filtered) < len(readings):
            return init, len(filtered), filtered, "NEG"

    return init, init, readings, None

# ==========================================
# MAIN
# ==========================================
def run(conllu_path, apertium_dir):
    stats = {k: 0 for k in [
        'sentences', 'align_failed', 'ambiguous',
        'fully_resolved', 'partially_resolved', 'untouched',
        'neg_full', 'neg_partial',
        'pos_strong_full', 'pos_strong_partial',
        'pos_weak_full', 'pos_weak_partial']}

    print("Reading CoNLL-U data...")
    data = list(read_conllu(conllu_path))
    texts = [t for t, _ in data]
    golds = [g for _, g in data]

    apt_batch = get_apertium_lus_batched(texts, apertium_dir)

    print("Aligning and applying zero-copula rules...")
    for i, g_tokens in enumerate(golds):
        if i >= len(apt_batch):
            break
        apt_lus = apt_batch[i]
        stats['sentences'] += 1

        g_forms, g_flat = filter_and_flatten([t['form'] for t in g_tokens], '_')
        a_forms, a_flat = filter_and_flatten([lu.wordform for lu in apt_lus], ' ')
        if not g_flat or not a_flat:
            continue

        sm = difflib.SequenceMatcher(None,
                                     [x['subword'] for x in g_flat],
                                     [x['subword'] for x in a_flat])
        aligned, pairs = True, []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                pairs.extend(zip(range(i1, i2), range(j1, j2)))
            else:
                aligned = False
                break
        if not aligned:
            stats['align_failed'] += 1
            continue

        comps = get_components([(g_flat[x]['orig_idx'], a_flat[y]['orig_idx']) for x, y in pairs])
        for g_idx, a_idx in comps:
            grp = [g_tokens[k] for k in g_idx]
            for ai in a_idx:
                lu = apt_lus[ai]
                if len(lu.readings) <= 1:
                    continue
                stats['ambiguous'] += 1
                init, final, _, rule = apply_mudt_rules(grp, lu)

                if final == 1:
                    stats['fully_resolved'] += 1
                    bucket = 'full'
                elif final < init:
                    stats['partially_resolved'] += 1
                    bucket = 'partial'
                else:
                    stats['untouched'] += 1
                    bucket = None

                if bucket:
                    if rule == 'NEG':
                        stats[f'neg_{bucket}'] += 1
                    elif rule == 'POS_strong':
                        stats[f'pos_strong_{bucket}'] += 1
                    elif rule == 'POS_weak':
                        stats[f'pos_weak_{bucket}'] += 1

    # ---- REPORT ----
    print("\n" + "=" * 55)
    print(" 🚀 ZERO-COPULA HYBRID RULE STATISTICS 🚀")
    print("=" * 55)
    print(f"Total Sentences Processed    : {stats['sentences']}")
    print(f"Sentences Skipped (Misalign) : {stats['align_failed']}")
    print("-" * 55)
    print(f"Ambiguous Words Encountered  : {stats['ambiguous']}")
    print(f"✅ Fully Resolved to 1       : {stats['fully_resolved']}")
    print(f"⚠️  Partially Reduced         : {stats['partially_resolved']}")
    print(f"❌ Untouched (For LLM)       : {stats['untouched']}")
    print("-" * 55)
    print(" BREAKDOWN BY RULE:")
    print(f"  Rule 1a NEGATIVE (modifier → drop <cop>):")
    print(f"     -> Fully resolved    : {stats['neg_full']}")
    print(f"     -> Partially reduced : {stats['neg_partial']}")
    print(f"  Rule 1b POSITIVE-strong (cop:zero head → keep <cop>):")
    print(f"     -> Fully resolved    : {stats['pos_strong_full']}")
    print(f"     -> Partially reduced : {stats['pos_strong_partial']}")
    print(f"  Rule 1b POSITIVE-weak (root+nominal → keep <cop>):")
    print(f"     -> Fully resolved    : {stats['pos_weak_full']}")
    print(f"     -> Partially reduced : {stats['pos_weak_partial']}")
    print("-" * 55)
    if stats['ambiguous'] > 0:
        rate = (stats['fully_resolved'] + stats['partially_resolved']) / stats['ambiguous'] * 100
        print(f"Rule-Based Intervention Rate : {rate:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_mudt_rule.py <conllu> <apertium_dir>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])