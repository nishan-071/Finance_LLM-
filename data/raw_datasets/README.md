# Raw datasets — untouched originals

Full, unmodified downloads of every open dataset this project uses. The sampled
files elsewhere in `data/` were cut from these by `scripts/01_collect_open_datasets.py`
(random seed 42, gates + contamination checks applied there). Nothing in this
folder is ever edited by hand; re-create it any time with
`python scripts/01b_export_raw_datasets.py`.

| File | Rows | Source |
|---|---|---|
| tat_llm_instructions_train.jsonl | 32,555 | huggingface.co/datasets/next-tat/tat-llm-instructions (CC-BY-4.0) |
| tat_llm_instructions_validation.jsonl | 4,136 | same |
| oasst2_train.jsonl | 128,575 | huggingface.co/datasets/OpenAssistant/oasst2 (Apache 2.0) |
| oasst2_validation.jsonl | 6,599 | same |
| finqa/train.json | 6,251 | github.com/czyssrs/FinQA (original authors' repo) |
| finqa/dev.json | 883 | same |
| finqa/test.json | 1,147 | same — this split is EVAL ONLY downstream |

Lineage (raw -> what the project actually uses):

- tat train (32,555) -> stratified sample, deduped, length-filtered, converted to
  chat messages -> 2,000 rows -> `data/train/mixins/tat_sample.jsonl`
- oasst2 train (128,575 messages) -> English conversation-opening best-ranked pairs
  (5,110) -> sample of 800 -> `data/train/mixins/oasst2_sample.jsonl`
- finqa test (1,147) -> copied whole -> `data/eval/finqa_test.jsonl` (never trained on)
- finqa train/dev sit here for reference only; the project does not train on them

Everything except this README and manifest.json is excluded from git (~500 MB).
