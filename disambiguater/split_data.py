import json, random
from collections import defaultdict

INPUT      = "uyghur_disambiguation_dataset.jsonl"
SEED       = 42
TRAIN, DEV = 0.8, 0.1   # test = remainder

random.seed(SEED)

# Group instances by sentence to prevent leakage
groups = defaultdict(list)
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        groups[d["sentence"]].append(d)

sentences = list(groups.keys())
random.shuffle(sentences)

n = len(sentences)
n_train = int(n * TRAIN)
n_dev   = int(n * DEV)

splits = {
    "train": sentences[:n_train],
    "dev":   sentences[n_train:n_train + n_dev],
    "test":  sentences[n_train + n_dev:],
}

for name, sents in splits.items():
    rows = [inst for s in sents for inst in groups[s]]
    with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:5s}: {len(sents):4d} sentences | {len(rows):5d} words")