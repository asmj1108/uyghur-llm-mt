import sys
import subprocess
import difflib
import io
from streamparser import parse_file

def filter_and_flatten(tokens, split_char):
    """
    1. Removes pure punctuation (tokens without alphanumeric characters).
    2. Splits tokens by `split_char` (space for Apertium, underscore for UyUDT).
    3. Returns the filtered original list AND a flattened list mapping each sub-word
       back to its original index.
    """
    orig_filtered = []
    flat_list = []

    for tok in tokens:
        if not any(c.isalnum() for c in tok):
            continue

        orig_filtered.append(tok)
        current_idx = len(orig_filtered) - 1 # Original index in the filtered list

        for subword in tok.split(split_char):
            if subword.strip():
                flat_list.append({
                    'orig_idx': current_idx,
                    'subword': subword
                })

    return orig_filtered, flat_list

def get_components(pairs):
    """
    Takes pairs of monotonically aligned (gold_idx, apt_idx) and groups them
    into contiguous N:M matching blocks.
    """
    components = []
    current_g = set()
    current_a = set()

    for g, a in pairs:
        # If both indices are new, start a new block
        if current_g and g not in current_g and a not in current_a:
            components.append((sorted(list(current_g)), sorted(list(current_a))))
            current_g = {g}
            current_a = {a}
        else:
            current_g.add(g)
            current_a.add(a)

    if current_g or current_a:
        components.append((sorted(list(current_g)), sorted(list(current_a))))

    return components

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
                if len(parts) >= 2 and '-' not in parts[0] and '.' not in parts[0]:
                    tokens.append(parts[1])

def get_apertium_tokens(text, apertium_dir):
    cmd = ["apertium", "-d", apertium_dir, "uig-tagger"]
    result = subprocess.run(cmd, input=text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return []
    output = result.stdout.decode('utf-8')
    stream = io.StringIO(output)
    return [lu.wordform for blank, lu in parse_file(stream, with_text=True) if lu is not None]

def test_alignment(conllu_path, apertium_dir):
    total_sentences = 0
    perfect_1_to_1 = 0
    resolved_nm = 0
    true_failed = 0

    for text, gold_tokens in read_conllu(conllu_path):
        if not text: continue

        apertium_tokens = get_apertium_tokens(text, apertium_dir)
        total_sentences += 1

        # Flatten CoNLL (split by _) and Apertium (split by space)
        gold_orig, gold_flat = filter_and_flatten(gold_tokens, '_')
        apt_orig, apt_flat = filter_and_flatten(apertium_tokens, ' ')

        if not gold_flat and not apt_flat:
            continue

        sm = difflib.SequenceMatcher(None, [x['subword'] for x in gold_flat], [x['subword'] for x in apt_flat])

        print_lines = []
        has_nm_mapping = False
        has_true_diff = False

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                pairs = list(zip(range(i1, i2), range(j1, j2)))
                # Extract index mappings
                for g_indices, a_indices in get_components([(gold_flat[x]['orig_idx'], apt_flat[y]['orig_idx']) for x, y in pairs]):
                    g_words = [gold_orig[i] for i in g_indices]
                    a_words = [apt_orig[i] for i in a_indices]

                    if len(g_words) == 1 and len(a_words) == 1:
                        print_lines.append(f"  [1:1] {g_words[0]} == {a_words[0]}")
                    else:
                        has_nm_mapping = True
                        print_lines.append(f"  [{len(g_words)}:{len(a_words)}] {' + '.join(g_words)} == {' + '.join(a_words)}  <-- MWE RESOLVED!")
            else:
                has_true_diff = True
                g_sw = [gold_flat[x]['subword'] for x in range(i1, i2)]
                a_sw = [apt_flat[y]['subword'] for y in range(j1, j2)]
                if tag == 'replace': print_lines.append(f"  [DIFF: MISMATCH] {g_sw} != {a_sw}")
                elif tag == 'delete': print_lines.append(f"  [DIFF: MISSING IN APT] {g_sw}")
                elif tag == 'insert': print_lines.append(f"  [DIFF: EXTRA IN APT] {a_sw}")

        # Update stats
        if has_true_diff:
            true_failed += 1
        elif has_nm_mapping:
            resolved_nm += 1
        else:
            perfect_1_to_1 += 1
            continue # Don't print boring 1:1 sentences

        # Print results if there's an N:M mapping or a true difference
        status = "⚠️ FAILED TO ALIGN" if has_true_diff else "✅ N:M SUCCESSFULLY RESOLVED"
        print(f"--- Sentence {total_sentences} | {status} ---")
        for line in print_lines:
            print(line)
        print("="*60)

    print("\n" + "#"*40)
    print("        ALIGNMENT MAPPING SUMMARY        ")
    print("#"*40)
    print(f"Total Sentences            : {total_sentences}")
    print(f"Perfect 1:1 Alignment      : {perfect_1_to_1}")
    print(f"N:M Mappings Resolved      : {resolved_nm}")
    print(f"True Misalignment (Errors) : {true_failed}")

    success_rate = ((perfect_1_to_1 + resolved_nm) / total_sentences) * 100
    print(f"Pipeline Compatibility Rate: {success_rate:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_alignment.py <path_to_conllu> <path_to_apertium_dir>")
        sys.exit(1)
    test_alignment(sys.argv[1], sys.argv[2])