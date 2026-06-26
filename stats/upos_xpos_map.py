import sys
from collections import defaultdict, Counter

def analyze_pos_mappings(conllu_file_path):
    """
    Parses a CoNLL-U dataset and creates frequency maps for UPOS and XPOS tags.
    """
    xpos_set = set()

    # Dictionaries to hold frequency counts of mappings
    xpos_to_upos_counts = defaultdict(Counter)
    upos_to_xpos_counts = defaultdict(Counter)

    total_tokens = 0
    missing_xpos = 0

    with open(conllu_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')

            # Skip multi-word token definitions (e.g., "1-2") or empty nodes (e.g., "1.1")
            if '-' in parts[0] or '.' in parts[0]:
                continue

            # CoNLL-U format: ID [0], FORM [1], LEMMA [2], UPOS [3], XPOS [4]
            if len(parts) >= 5:
                upos = parts[3]
                xpos = parts[4]

                total_tokens += 1

                if xpos == '_':
                    missing_xpos += 1
                else:
                    xpos_set.add(xpos)

                xpos_to_upos_counts[xpos][upos] += 1
                upos_to_xpos_counts[upos][xpos] += 1

    # --- OUTPUT REPORT ---
    print("========================================")
    print(f"Dataset Statistics:")
    print(f"Total valid tokens: {total_tokens}")
    print(f"Tokens with missing XPOS ('_'): {missing_xpos}")
    print(f"Unique XPOS tags found: {len(xpos_set)}")
    print(f"XPOS Set: {sorted(list(xpos_set))}")
    print("========================================\n")

    print("--- XPOS to UPOS Mapping Frequencies ---")
    print("(Use this to find instances where an XPOS is used for strange UPOS tags)")
    for xpos in sorted(xpos_to_upos_counts.keys()):
        if xpos == '_': continue
        mappings = xpos_to_upos_counts[xpos]
        total_uses = sum(mappings.values())
        print(f"XPOS '{xpos}' (Total: {total_uses}) maps to:")
        for upos, count in mappings.most_common():
            percentage = (count / total_uses) * 100
            print(f"   -> {upos}: {count} times ({percentage:.1f}%)")
        print()

    print("--- UPOS to XPOS Mapping Frequencies ---")
    for upos in sorted(upos_to_xpos_counts.keys()):
        if upos == '_': continue
        mappings = upos_to_xpos_counts[upos]
        total_uses = sum(mappings.values())
        print(f"UPOS '{upos}' (Total: {total_uses}) derived from:")
        for xpos, count in mappings.most_common():
            percentage = (count / total_uses) * 100
            print(f"   -> XPOS '{xpos}': {count} times ({percentage:.1f}%)")
        print()

if __name__ == "__main__":
    conllu_file = "../uyudt_all.conllu"
    analyze_pos_mappings(conllu_file)