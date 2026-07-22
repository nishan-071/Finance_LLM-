"""Turn verified facts into training flashcards: many reworded QUESTIONS per
fact, written by a local open model (the base Ministral itself -- keeps the
no-frontier-API rule airtight). ANSWERS are assembled deterministically from
the fact sheet, never model-written, so no number can drift in training data.

Per fact: ~20-30 candidate questions generated, deduplicated, last N held out
as the dev set (never trained -- the honest measure of learning).

Run on a GPU box (infer venv) from the repo root:
    python scripts/09_generate_paraphrases.py
Takes ~1.5-2.5 hours for 62 facts. Restart-safe: already-processed facts are
skipped on rerun.
"""
import argparse
import csv
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = """You write exam questions for a finance quiz.

Here is a verified fact from JPMorgan Chase's 2025 Form 10-K:
"{fact}"

Write {n} different questions a person might ask whose correct answer is exactly this fact.
Vary the phrasing, formality and angle: ask for the value directly, ask about the change
versus the prior year, ask casually, ask formally, use abbreviations, use full terms.
Do not include the answer. Output ONLY a numbered list of the {n} questions."""


def parse_questions(text):
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[\.\)\:]?\s*(.+)", line)
        if m:
            q = m.group(1).strip().strip('"').strip("*").strip()
            if len(q) > 15:
                if not q.endswith("?"):
                    q += "?"
                out.append(q)
    return out


def dedup(questions):
    seen, uniq = set(), []
    for q in questions:
        key = re.sub(r"[^a-z0-9]", "", q.lower())
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--facts", default="data/facts/fact_sheet.csv")
    ap.add_argument("--train-out", default="data/train/fact_cards_train.jsonl")
    ap.add_argument("--dev-out", default="data/train/fact_cards_dev.jsonl")
    ap.add_argument("--calls", type=int, default=3, help="generation calls per fact")
    ap.add_argument("--per-call", type=int, default=10, help="questions requested per call")
    ap.add_argument("--holdout", type=int, default=3, help="dev questions held out per fact")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.facts, encoding="utf-8") as f:
        facts = list(csv.DictReader(f))

    done = set()
    for path in (args.train_out, args.dev_out):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done |= {json.loads(l)["id"].rsplit("-", 1)[0] for l in f if l.strip()}
    if done:
        print(f"[resuming: {len(done)} facts already processed]")

    torch.manual_seed(args.seed)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quant)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quant)

    citation = "per JPMorganChase's 2025 Form 10-K, filed February 13, 2026"
    train_f = open(args.train_out, "a", encoding="utf-8")
    dev_f = open(args.dev_out, "a", encoding="utf-8")

    for idx, fact in enumerate(facts, 1):
        fid, stmt = fact["fact_id"], fact["fact_statement"]
        if fid in done:
            continue
        questions = []
        for _ in range(args.calls):
            enc = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(fact=stmt, n=args.per_call)}],
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt").to(model.device)
            gen = model.generate(**enc, max_new_tokens=600, do_sample=True,
                                 temperature=0.9, top_p=0.95)
            questions += parse_questions(
                tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        questions = dedup(questions)
        if len(questions) < 6:
            print(f"[warn] {fid}: only {len(questions)} usable questions, skipping holdout")
        dev_qs = questions[-args.holdout:] if len(questions) > args.holdout + 5 else []
        train_qs = [q for q in questions if q not in dev_qs]

        # Deterministic answers built from the verified sheet -- never generated.
        answers = [f"{stmt} ({citation}).",
                   f"Per the 2025 Form 10-K (filed February 13, 2026): {stmt}."]
        for i, q in enumerate(train_qs):
            rec = {"messages": [{"role": "user", "content": q},
                                {"role": "assistant", "content": answers[i % 2]}],
                   "source": "jpm_facts_2026", "id": f"{fid}-p{i:02d}"}
            train_f.write(json.dumps(rec) + "\n")
        for i, q in enumerate(dev_qs):
            rec = {"messages": [{"role": "user", "content": q},
                                {"role": "assistant", "content": answers[i % 2]}],
                   "source": "jpm_facts_2026_dev", "id": f"{fid}-d{i:02d}"}
            dev_f.write(json.dumps(rec) + "\n")
        train_f.flush(); dev_f.flush()
        print(f"[{idx}/{len(facts)}] {fid}: {len(train_qs)} train + {len(dev_qs)} dev cards")

    train_f.close(); dev_f.close()
    print(f"\nDone -> {args.train_out} + {args.dev_out}")


if __name__ == "__main__":
    main()
