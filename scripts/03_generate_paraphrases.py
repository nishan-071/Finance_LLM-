"""Turn the frozen fact sheet into the primary training corpus.

For every fact: a LOCAL model (Ollama or any OpenAI-compatible endpoint) writes
N differently-worded questions; the answer stays canonical — built from the fact
row itself, citation included — so the model can't hallucinate answers into our
training data. A few paraphrases per fact are held out as the dev set.

Usage (Ollama running locally):
    python scripts/03_generate_paraphrases.py --model ministral-3:14b
    python scripts/03_generate_paraphrases.py --model X --n 20 --holdout 3

Rule 2 reminder: --base-url must point at a LOCAL open model. Never a frontier API.

Writes:
    data/train/own_corpus.jsonl   ({messages, source: "own_sec", id})
    data/eval/dev_holdout.jsonl   (held-out paraphrases, never trained)
"""
import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def chat(base_url, model, prompt, temperature=0.9):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(ROOT / "data" / "facts" / "fact_sheet.csv"))
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20, help="question rewordings per fact")
    ap.add_argument("--holdout", type=int, default=3, help="paraphrases held out per fact for the dev set")
    args = ap.parse_args()

    facts = list(csv.DictReader(open(args.facts, encoding="utf-8-sig")))
    facts = [f for f in facts if "SAMPLE" not in f["fact_statement"]]
    if not facts:
        raise SystemExit("fact sheet has no real rows yet (only SAMPLE rows)")
    print(f"{len(facts)} facts loaded from {args.facts}")

    train, dev = [], []
    for f in facts:
        cite = f' (per the {f["source_doc"]} filed {f["filed_date"]})' if f.get("filed_date") else ""
        answer = f["fact_statement"].strip().rstrip(".") + cite + "."
        prompt = (
            f"Here is one verified fact from a company filing:\n\"{f['fact_statement']}\"\n\n"
            f"Write {args.n} different ways a finance professional might ask a question whose answer is exactly this fact. "
            "Vary vocabulary, sentence structure, and angle (direct value, year-over-year change, comparison). "
            "One question per line, numbered. Questions only, no answers."
        )
        raw = chat(args.base_url, args.model, prompt)
        qs, seen = [], set()
        for line in raw.splitlines():
            q = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            if q.endswith("?") and len(q) > 15 and norm(q) not in seen:
                seen.add(norm(q))
                qs.append(q)
        if len(qs) < 5:
            print(f"WARN {f['fact_id']}: only {len(qs)} usable questions, skipping fact")
            continue
        random.shuffle(qs)
        held, kept = qs[: args.holdout], qs[args.holdout:]
        for i, q in enumerate(kept):
            train.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ],
                "source": "own_sec", "id": f"{f['fact_id']}-{i}",
            })
        for i, q in enumerate(held):
            dev.append({"fact_id": f["fact_id"], "question": q, "expected": answer})
        print(f"{f['fact_id']}: {len(kept)} train + {len(held)} dev")

    out_train = ROOT / "data" / "train" / "own_corpus.jsonl"
    out_dev = ROOT / "data" / "eval" / "dev_holdout.jsonl"
    with open(out_train, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dev, "w", encoding="utf-8") as fh:
        for r in dev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(train)} training pairs -> {out_train.relative_to(ROOT)}")
    print(f"{len(dev)} held-out dev questions -> {out_dev.relative_to(ROOT)} (never train on this)")


if __name__ == "__main__":
    main()
