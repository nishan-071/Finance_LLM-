"""Record the "before": run every locked exam question against a model and log
answers verbatim with timestamps. Run once against base Ministral 3 14B the day
the exam is locked; the same script produces the "after" log post-training.

Works against any OpenAI-compatible endpoint (Ollama local, vLLM on RunPod):
    python scripts/05_before_log.py --model ministral-3:14b --tag base
    python scripts/05_before_log.py --model finance-model --tag tuned

Writes data/eval/log_{tag}.jsonl - one row per question, answer untouched.
"""
import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chat(base_url, model, question):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.15,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=str(ROOT / "data" / "eval" / "exam.csv"))
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="base | tuned | anything descriptive")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.exam, encoding="utf-8-sig")))
    rows = [r for r in rows if "SAMPLE" not in r["question"]]
    if not rows:
        raise SystemExit("exam file has no real questions yet (only SAMPLE rows)")

    out = ROOT / "data" / "eval" / f"log_{args.tag}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            answer = chat(args.base_url, args.model, r["question"])
            fh.write(json.dumps({
                "qid": r["qid"], "type": r["type"], "question": r["question"],
                "expected": r.get("expected_answer", ""),
                "model": args.model, "answer": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['qid']} logged")
    print(f"\n{len(rows)} answers -> {out.relative_to(ROOT)}")
    print("This file is evidence. Do not edit it, ever.")


if __name__ == "__main__":
    main()
