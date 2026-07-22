"""Export the untouched FULL originals into data/raw_datasets/.

These are the sources that the sampled files in data/train/mixins/ and
data/eval/ were cut from (by scripts/01_collect_open_datasets.py, seed 42).
Kept in the project so the raw -> sample separation is visible and auditable
without digging into the Hugging Face cache. Big files: excluded from git.
"""
import json
import urllib.request
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_datasets"
(RAW / "finqa").mkdir(parents=True, exist_ok=True)


def save_jsonl(rows, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    mb = path.stat().st_size / 1e6
    print(f"{path.relative_to(ROOT)}  {n} rows  {mb:.1f} MB")
    return n


manifest = {}

# tat-llm-instructions: both splits, untouched
tat = load_dataset("next-tat/tat-llm-instructions")
manifest["tat_llm_instructions_train"] = save_jsonl(tat["train"], RAW / "tat_llm_instructions_train.jsonl")
manifest["tat_llm_instructions_validation"] = save_jsonl(tat["validation"], RAW / "tat_llm_instructions_validation.jsonl")

# oasst2: both splits, full message trees with metadata
oa = load_dataset("OpenAssistant/oasst2")
manifest["oasst2_train"] = save_jsonl(oa["train"], RAW / "oasst2_train.jsonl")
manifest["oasst2_validation"] = save_jsonl(oa["validation"], RAW / "oasst2_validation.jsonl")

# FinQA: all three splits from the original authors' repository
for split in ("train", "dev", "test"):
    url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
    with urllib.request.urlopen(url) as resp:
        rows = json.load(resp)
    out = RAW / "finqa" / f"{split}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest[f"finqa_{split}"] = len(rows)
    print(f"{out.relative_to(ROOT)}  {len(rows)} rows  {out.stat().st_size / 1e6:.1f} MB")

(RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("\nmanifest ->", (RAW / "manifest.json").relative_to(ROOT))
