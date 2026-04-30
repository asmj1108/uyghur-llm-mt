import sys
import subprocess
import difflib
import io
import re
from streamparser import parse_file, reading_to_string

# Map Apertium base tags to Universal Dependencies UPOS tags
# (Derived from uig-feats.tsv rules)
APT_TO_UPOS = {
    'n': 'NOUN', 'np': 'PROPN', 'v': 'VERB', 'vaux': 'AUX', 'cop': 'AUX',
    'adj': 'ADJ', 'adv': 'ADV', 'post': 'ADP', 'postadv': 'ADV', 'prn': 'PRON',
    'det': 'DET', 'num': 'NUM', 'cnjcoo': 'CCONJ', 'cnjsub': 'SCONJ', 'cnjadv': 'SCONJ',
    'ij': 'INTJ', 'mod': 'PART', 'mod_ass': 'PART', 'mod_emo': 'PART', 'emph': 'PART',
    'qst': 'PART', 'cm': 'PUNCT', 'lpar': 'PUNCT', 'rpar': 'PUNCT', 'lquot': 'PUNCT',
    'rquot': 'PUNCT', 'sent': 'PUNCT', 'guio': 'PUNCT', 'abbr': 'NOUN', 'sym': 'SYM', 'x': 'X'
}

def extract_upos_from_reading(reading_str):
    """
    Extracts all Apertium tags from a reading (e.g., '<n><loc>')
    and translates them to a set of UPOS tags {NOUN}.
    Handles joined encodings like <n><loc>+<cop> -> {NOUN, AUX}.
    """
    tags = re.findall(r'<([^>]+)>', reading_str)
    return {APT_TO_UPOS[t] for t in tags if t in APT_TO_UPOS}

def read_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = ""
        tokens = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    yield text, tokens
                text = ""
                tokens = []
                continue
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            elif line.startswith("#"):
                continue
            else:
                parts = line.split("\t")
                if len(parts) >= 4 and '-' not in parts[0] and '.' not in parts[0]:
                    tokens.append({'form': parts[1], 'upos': parts[3]})

def get_apertium_data(text, apertium_dir):
    """Returns a list of dicts: {'wordform': str, 'readings': [str, str...]}"""
    cmd = ["apertium", "-d", apertium_dir, "uig-tagger"]
    result = subprocess.run(cmd, input=text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return []

    stream = io.StringIO(result.stdout.decode('utf-8'))
    apt_data = []
    for blank, lu in parse_file(stream, with_text=True):
        if lu is not None:
            readings_strs = [reading_to_string(r) for r in lu.readings]
            apt_data.append({'wordform': lu.wordform, 'readings': readings_strs})
    return apt_data

def filter_and_flatten_gold(tokens):
    orig_filtered, flat_list = [], []
    for g in tokens:
        if not any(c.isalnum() for c in g['form']): continue
        orig_filtered.append(g)
        idx = len(orig_filtered) - 1
        for subword in g['form'].split('_'):
            if subword.strip():
                flat_list.append({'orig_idx': idx, 'subword': subword})
    return orig_filtered, flat_list

def filter_and_flatten_apt(tokens):
    orig_filtered, flat_list = [], []
    for a in tokens:
        if not any(c.isalnum() for c in a['wordform']): continue
        orig_filtered.append(a)
        idx = len(orig_filtered) - 1
        for subword in a['wordform'].split(' '):
            if subword.strip():
                flat_list.append({'orig_idx': idx, 'subword': subword})
    return orig_filtered, flat_list

def get_components(pairs):
    components, current_g, current_a = [], set(), set()
    for g, a in pairs:
        if current_g and g not in current_g and a not in current_a:
            components.append((sorted(list(current_g)), sorted(list(current_a))))
            current_g, current_a = {g}, {a}
        else:
            current_g.add(g); current_a.add(a)
    if current_g or current_a:
        components.append((sorted(list(current_g)), sorted(list(current_a))))
    return components

def run_validation_test(conllu_files, apertium_dir):
    ambiguous_before = 0
    ambiguous_after = 0
    successfully_disambiguated = 0
    overfiltered_to_zero = 0  # CRITICAL: Tracks cases where the filter deleted ALL readings

    for c_file in conllu_files:
        print(f"Processing: {c_file}...")
        for text, gold_tokens in read_conllu(c_file):
            apt_tokens = get_apertium_data(text, apertium_dir)

            g_orig, g_flat = filter_and_flatten_gold(gold_tokens)
            a_orig, a_flat = filter_and_flatten_apt(apt_tokens)

            if not g_flat or not a_flat: continue

            sm = difflib.SequenceMatcher(None, [x['subword'] for x in g_flat], [x['subword'] for x in a_flat])

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    pairs = [(g_flat[x]['orig_idx'], a_flat[y]['orig_idx']) for x, y in zip(range(i1, i2), range(j1, j2))]

                    for g_indices, a_indices in get_components(pairs):
                        # 1. Collect Set of Gold UPOS tags for this alignment block
                        gold_upos_set = {g_orig[i]['upos'] for i in g_indices}

                        # 2. Evaluate each aligned Apertium unit
                        # (Use set() because N:M might hit the same Apertium unit multiple times)
                        for a_idx in set(a_indices):
                            readings = a_orig[a_idx]['readings']

                            if len(readings) > 1:
                                ambiguous_before += 1
                                valid_readings = 0

                                # Apply UPOS Filter
                                for r in readings:
                                    apt_upos_set = extract_upos_from_reading(r)
                                    # If the Apertium reading contains a UPOS that matches our Gold CoNLL tags
                                    if apt_upos_set.intersection(gold_upos_set):
                                        valid_readings += 1

                                # Tally Results
                                if valid_readings > 1:
                                    ambiguous_after += 1
                                elif valid_readings == 1:
                                    successfully_disambiguated += 1
                                elif valid_readings == 0:
                                    overfiltered_to_zero += 1

    # --- PRINT FINAL REPORT ---
    print("\n" + "="*50)
    print("      UPOS FILTER VALIDATION RESULT")
    print("="*50)
    if ambiguous_before == 0:
        print("No ambiguous words found after CG.")
        return

    reduction = ((ambiguous_before - ambiguous_after) / ambiguous_before) * 100

    print(f"Total Words Ambiguous Before Filter : {ambiguous_before}")
    print(f"Total Words Ambiguous After Filter  : {ambiguous_after}")
    print(f"Reduction in Ambiguity              : {reduction:.2f}%\n")

    print(f"Outcomes of Ambiguous Words:")
    print(f"  ➜ Still Ambiguous         : {ambiguous_after}")
    print(f"  ➜ Fully Disambiguated (1) : {successfully_disambiguated}")
    print(f"  ➜ Overfiltered (0 left)   : {overfiltered_to_zero}")
    print("\n*Note: 'Overfiltered' means the filter deleted all readings. This happens")
    print("if the Apertium tag mapping is strictly missing or the gold UPOS is wrong.")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_upos_filter.py <apertium_dir> <conllu_file1> [conllu_file2 ...]")
        sys.exit(1)

    apertium_directory = sys.argv[1]
    conllu_paths = sys.argv[2:]

    run_validation_test(conllu_paths, apertium_directory)