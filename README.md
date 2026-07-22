# Finance LLM Fine-Tune

Teach an open model (Ministral 3 14B) facts from 2026 SEC filings it provably
doesn't know, and demo the before/after live. Closed-book — no RAG in the demo
(decision: Manish, July 2026). Full context in `docs/`:

- `docs/Finance_LLM_Team_Playbook.docx` — the how, step by step
- `docs/Finance_LLM_Kickoff_Brief.docx` — verified decisions + who starts what
- `docs/provenance_sheet.md` — every dataset, license, and verdict (client counsel will ask)

## Layout

```
data/raw_filings/   downloaded 10-Ks (not committed)
data/raw_datasets/  untouched full originals of every open dataset (not committed;
                    manifest + README are; recreate via scripts/01b)
data/facts/         the fact sheet: 50-150 atomic facts w/ filing, page, date
data/train/         train.jsonl (paraphrased QA) + mix-ins (cut from raw_datasets)
data/eval/          the locked 100-question exam + before-log + dev set
scripts/            download, parse, paraphrase loop, eval scoring
notebooks/          Colab/Unsloth training notebooks
app/                Ollama modelfile, demo side-by-side UI
```

## The rules (short form)

1. Exam written and locked before any training. No edits after.
2. No frontier-API output anywhere in the training path (paraphrase generator = local model).
3. Never train on: Sujet-177k, Finance-Instruct-500k, financial_phrasebank, FinanceBench,
   XBRL_analysis (no license tag), Fino1 (GPT-4o provenance), SmolTalk (Llama provenance),
   Tulu 3 SFT (NC + GPT-4 portions). See provenance sheet.
4. Early stopping at best dev-set score. More epochs = more hallucination.
5. Stop RunPod pods when idle (~$82/week if forgotten).
6. Week-0 market gate: no named demo audience booked = no paid GPU time.

## Recipe at a glance

Ministral 3 14B (Unsloth 4-bit) · QLoRA r=32 alpha=64 all-linear · lr 2e-4 cosine ·
2-3 epochs, early stop · rehearse on 8B in free Colab first · pass bar at week 4:
>=70% held-out paraphrases, refusals unchanged, general ability within 2 pts.
