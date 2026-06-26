import json, numpy as np, re

T_OPEN, T_CLOSE = "<t>", "</t>"

def mark(s, w, occ=0):
    p = re.compile(rf"(?<!\S){re.escape(w.strip())}(?!\S)")
    ms = list(p.finditer(s))
    if ms:
        m = ms[occ] if occ < len(ms) else ms[0]
        return s[:m.start()] + f"{T_OPEN} {w} {T_CLOSE}" + s[m.end():]
    return s

def prompt(r):
    sent = mark(r["sentence"], r["target_word"], r.get("target_occurrence", 0))
    cands = "\n".join(f"{i+1}. {c}" for i, c in enumerate(r["candidates"]))
    return (f"Disambiguate the marked Uyghur word by choosing the analysis whose "
            f"features are ALL correct.\nSentence: {sent}\nWord: {r['target_word']}\n"
            f"Candidates:\n{cands}")

src_len, tgt_len, ncand = [], [], []
for split in ["train", "dev", "test"]:
    for line in open(f"../{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        # ByT5 length is EXACTLY the number of UTF-8 bytes + 1 (for the EOS token)
        src_len.append(len(prompt(r).encode("utf-8")) + 1)
        tgt_len.append(len(r["target_text"].encode("utf-8")) + 1)
        ncand.append(len(r["candidates"]))

def pct(a, ps=(50, 90, 95, 99, 99.5, 100)):
    a = np.array(a)
    return {p: int(np.percentile(a, p)) for p in ps}

print("SOURCE token len :", pct(src_len))
print("TARGET token len :", pct(tgt_len))
print("num candidates   :", pct(ncand))