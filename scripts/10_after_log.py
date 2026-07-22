"""After-log: run the LOCKED exam through the FINE-TUNED model and record every
answer with a timestamp, so it lines up one-for-one against the before-log.
Same stack, same temperature 0 -- the only change is the adapter is loaded.

Run on a GPU box (infer venv) from the repo root:
    python scripts/10_after_log.py --adapter /path/to/adapter
"""
import argparse
import csv
import datetime
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--adapter", required=True, help="path to the trained adapter folder")
    ap.add_argument("--exam", default="data/eval/exam_locked.csv")
    ap.add_argument("--out", default="data/eval/after_log.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=250)
    args = ap.parse_args()

    with open(args.exam, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["qid"] for line in f if line.strip()}
        print(f"[resuming: {len(done)} answers already logged]")

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
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"[loaded adapter: {args.adapter}]")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(questions, 1):
            if row["qid"] in done:
                continue
            enc = tok.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt").to(model.device)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)  # temperature 0
            answer = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
            rec = {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "qid": row["qid"], "type": row["type"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model": args.model + " + adapter", "phase": "after_training",
                "answer": answer,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i}/{len(questions)}] {row['qid']} ({row['type']}): {answer[:90]}...")

    print(f"\nAfter-log complete -> {args.out}")


if __name__ == "__main__":
    main()
