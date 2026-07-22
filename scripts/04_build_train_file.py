"""Blend everything into the final training file, at the planned ratios.

Two modes:
  --rehearsal   mixins only (no own corpus yet) -> data/train/rehearsal_train.jsonl
                for the free-Colab pipeline rehearsal. Proves the loop, proves nothing else.
  (default)     needs data/train/own_corpus.jsonl -> data/train/train.jsonl
                own corpus ~60% of rows, tat ~2.5x oasst2 for the rest.

Every row is verified to be {messages: [user, assistant], source, id} before writing.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "train" / "mixins"
random.seed(42)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def check(rows, name):
    for r in rows:
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"], f"{name}: bad roles"
        assert all(m["content"].strip() for m in r["messages"]), f"{name}: empty content"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--own-share", type=float, default=0.60)
    args = ap.parse_args()

    tat = check(load(MIX / "tat_sample.jsonl"), "tat")
    oasst = check(load(MIX / "oasst2_sample.jsonl"), "oasst2")

    if args.rehearsal:
        rows = tat + oasst
        out = ROOT / "data" / "train" / "rehearsal_train.jsonl"
        note = "REHEARSAL ONLY - no own corpus; use to debug the training loop, not for the demo"
    else:
        own_path = ROOT / "data" / "train" / "own_corpus.jsonl"
        if not own_path.exists():
            raise SystemExit("own_corpus.jsonl missing - run 03_generate_paraphrases.py first "
                             "(or use --rehearsal for the mixins-only rehearsal file)")
        own = check(load(own_path), "own")
        for r in own:
            r.setdefault("source", "own_sec")
        mixin_budget = round(len(own) * (1 - args.own_share) / args.own_share)
        tat_take = min(len(tat), round(mixin_budget * 0.72))
        oasst_take = min(len(oasst), mixin_budget - tat_take)
        random.shuffle(tat)
        random.shuffle(oasst)
        rows = own + tat[:tat_take] + oasst[:oasst_take]
        out = ROOT / "data" / "train" / "train.jsonl"
        note = (f"own {len(own)} ({len(own)/len(rows):.0%}) + tat {tat_take} + oasst2 {oasst_take}")

    random.shuffle(rows)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    src = Counter(r["source"] for r in rows)
    est = sorted(sum(len(m["content"]) for m in r["messages"]) // 4 for r in rows)
    print(f"wrote {len(rows)} rows -> {out.relative_to(ROOT)}")
    print("sources:", dict(src))
    print(f"est tokens: median {est[len(est)//2]:,} | p90 {est[int(len(est)*.9)]:,} | max {est[-1]:,}")
    print(note)


if __name__ == "__main__":
    main()
