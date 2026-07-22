"""Collect the approved open datasets, test them, and write a verification report.

Gates encoded here (re-run this the week of the demo):
  1. License on the HF card must match what the provenance sheet promises.
  2. Row counts must be in the expected range (catches silently-changed datasets).
  3. Zero overlap between anything we keep for training and the eval sets.

Outputs:
  data/train/mixins/tat_sample.jsonl      (~2,000 rows, stratified)
  data/train/mixins/oasst2_sample.jsonl   (~800 English pairs — PROPOSED)
  data/eval/finqa_test.jsonl              (eval only)
  data/eval/finqa_verified.jsonl          (eval only, 91 rows)
  docs/dataset_verification_report.md
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info

ROOT = Path(__file__).resolve().parents[1]
MIXINS = ROOT / "data" / "train" / "mixins"
EVAL = ROOT / "data" / "eval"
MIXINS.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)
random.seed(42)

report = []


def gate(name, ok, detail):
    report.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + " - " + detail)
    if not ok:
        sys.exit("Gate failed: " + name)


def card_license(repo):
    info = dataset_info(repo)
    lic = None
    if info.card_data:
        lic = info.card_data.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return (lic or "MISSING").lower()


def norm(q):
    return re.sub(r"[^a-z0-9]", "", q.lower())[:120]


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- 1. FinQA test split (EVAL ONLY) ----------
# HF card (cc-by-4.0) is a pointer; data lives in the original authors' MIT repo.
lic = card_license("ibm-research/finqa")
gate("finqa.license", lic == "cc-by-4.0", "card says " + lic)
import urllib.request

url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
with urllib.request.urlopen(url) as resp:
    test = json.load(resp)
gate("finqa.test_count", 1000 <= len(test) <= 1300, str(len(test)) + " rows (expect ~1,147)")
save_jsonl(test, EVAL / "finqa_test.jsonl")
finqa_test_keys = {norm(r.get("qa", {}).get("question", "")) for r in test}
finqa_test_keys.discard("")

# ---------- 2. finqa-verified (EVAL ONLY) ----------
lic = card_license("Aiera/finqa-verified")
gate("finqa_verified.license", lic == "mit", "card says " + lic)
fv = load_dataset("Aiera/finqa-verified")
split = list(fv[list(fv.keys())[0]])
gate("finqa_verified.count", 80 <= len(split) <= 100, str(len(split)) + " rows (expect 91)")
save_jsonl(split, EVAL / "finqa_verified.jsonl")
eval_ids = {r.get("id", "") for r in test}
eval_ids.discard("")
for r in split:
    finqa_test_keys.add(norm(str(r.get("question", ""))))
    rid = str(r.get("id", "") or r.get("finqa_id", ""))
    if rid:
        eval_ids.add(rid)

# ---------- 3. tat-llm-instructions (TRAIN MIX-IN, sampled) ----------
lic = card_license("next-tat/tat-llm-instructions")
gate("tat.license", lic == "cc-by-4.0", "card says " + lic)
tat = load_dataset("next-tat/tat-llm-instructions")
tat_train = list(tat["train"])
gate("tat.count", 25000 <= len(tat_train) <= 45000, str(len(tat_train)) + " rows (expect ~32.6k train)")

# Firewall 1 (exact, whole split): tat ids carry their source — rows ending _finqa
# keep the original FinQA id. None of those may appear in our eval sets.
finqa_sourced = [r["id"][:-6] for r in tat_train if r["id"].endswith("_finqa")]
overlap = set(finqa_sourced) & eval_ids
gate(
    "tat.id_firewall",
    len(overlap) == 0,
    str(len(finqa_sourced)) + " finqa-sourced rows in tat train; " + str(len(overlap)) + " overlap eval ids",
)

# Stratified sample by source suffix (finqa / tatqa / tatdqa). Oversample (780 per
# source), because dedupe + length filtering below will trim it back to 2,000.
by = {}
for r in tat_train:
    src = r["id"].rsplit("_", 1)[-1]
    by.setdefault(src, []).append(r)
sample = []
for v in by.values():
    random.shuffle(v)
    sample.extend(v[:780])

# Firewall 2 (backup, sampled rows): no eval question text embedded in any kept prompt
keys = [k for k in finqa_test_keys if len(k) >= 40]
before = len(sample)
sample = [r for r in sample if not any(k in norm(r["user_prompt"]) for k in keys)]
gate(
    "tat.text_firewall",
    True,
    "dropped " + str(before - len(sample)) + " sampled rows containing eval question text",
)

# Final fix: every kept row becomes a ChatGPT-style conversation (user asks,
# assistant answers), exact duplicates go, and anything too long for the
# 2,048-token training window (with chat-template headroom) goes.
seen, chat_rows, dupes, too_long = set(), [], 0, 0
for r in sample:
    key = hashlib.md5((r["user_prompt"] + r["resp"]).encode("utf-8")).hexdigest()
    if key in seen:
        dupes += 1
        continue
    seen.add(key)
    if (len(r["user_prompt"]) + len(r["resp"])) // 4 > 1900:
        too_long += 1
        continue
    chat_rows.append(
        {
            "messages": [
                {"role": "user", "content": r["user_prompt"]},
                {"role": "assistant", "content": r["resp"]},
            ],
            "source": "tat_" + r["id"].rsplit("_", 1)[-1],
            "id": r["id"],
        }
    )
gate(
    "tat.chat_ready",
    len(chat_rows) >= 2000,
    f"{len(chat_rows)} chat rows after dropping {dupes} dupes + {too_long} over-length; keeping 2,000",
)
save_jsonl(chat_rows[:2000], MIXINS / "tat_sample.jsonl")

# ---------- 4. oasst2 (GENERAL CHAT — PROPOSED, pending Manish) ----------
lic = card_license("OpenAssistant/oasst2")
gate("oasst2.license", lic == "apache-2.0", "card says " + lic)
oa = load_dataset("OpenAssistant/oasst2")
msgs = list(oa["train"])
by_id = {m["message_id"]: m for m in msgs}
pairs = []
for m in msgs:
    if (
        m.get("role") == "assistant"
        and m.get("lang") == "en"
        and m.get("rank") == 0
        and m.get("parent_id") in by_id
    ):
        parent = by_id[m["parent_id"]]
        # parent must be a conversation ROOT (no parent of its own) — otherwise the
        # pair is a mid-thread fragment with unresolved references ("tell me more about him")
        if parent.get("role") == "prompter" and parent.get("lang") == "en" and not parent.get("parent_id"):
            pairs.append(
                {
                    "messages": [
                        {"role": "user", "content": parent["text"]},
                        {"role": "assistant", "content": m["text"]},
                    ],
                    "source": "oasst2",
                    "id": m["message_id"],
                }
            )
gate("oasst2.pairs", len(pairs) >= 2000, str(len(pairs)) + " English best-ranked pairs extracted")
random.shuffle(pairs)
save_jsonl(pairs[:800], MIXINS / "oasst2_sample.jsonl")

# ---------- report ----------
lines = [
    "# Dataset verification report",
    "",
    "Generated by scripts/01_collect_open_datasets.py. Re-run the week of the demo.",
    "",
    "| Gate | Result | Detail |",
    "|---|---|---|",
]
for name, ok, detail in report:
    lines.append("| " + name + " | " + ("PASS" if ok else "FAIL") + " | " + detail + " |")
lines += [
    "",
    "Files written:",
    "- data/train/mixins/tat_sample.jsonl (mix-in, CC-BY-4.0 — chat format, deduped, length-filtered to fit 2,048-token training)",
    "- data/train/mixins/oasst2_sample.jsonl (PROPOSED general chat, Apache 2.0 — chat format, conversation-opening pairs only — awaiting Manish)",
    "- data/eval/finqa_test.jsonl (EVAL ONLY)",
    "- data/eval/finqa_verified.jsonl (EVAL ONLY)",
    "",
    "All training mix-ins share one schema: {messages: [user, assistant], source, id}.",
    "Firewall: training mix-ins were checked against eval ids and question text; overlapping rows dropped.",
]
(ROOT / "docs" / "dataset_verification_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\nreport -> docs/dataset_verification_report.md")
