"""Download a company's latest 10-K from SEC EDGAR and parse it to clean text.

Usage (venv):
    python scripts/02_download_filing.py TICKER
    python scripts/02_download_filing.py TICKER --min-filed 2026-01-01

Writes:
    data/raw_filings/{TICKER}_10-K_{filed}.html   (original filing document)
    data/raw_filings/{TICKER}_10-K_{filed}.md     (Docling-parsed markdown)

The --min-filed guard exists because the whole demo depends on post-cutoff
facts: a 10-K filed before 2026 is useless to us, so the script refuses it
unless you override.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_filings"
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "SamniLabs research kkharel1234@gmail.com"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--form", default="10-K")
    ap.add_argument("--min-filed", default="2026-01-01",
                    help="refuse filings older than this (post-cutoff rule)")
    args = ap.parse_args()
    ticker = args.ticker.upper()

    # ticker -> CIK
    tickers = get_json("https://www.sec.gov/files/company_tickers.json")
    cik = None
    for row in tickers.values():
        if row["ticker"].upper() == ticker:
            cik = int(row["cik_str"])
            company = row["title"]
            break
    if cik is None:
        sys.exit(f"ticker {ticker} not found in SEC company list")
    print(f"{ticker} = CIK {cik} ({company})")

    # latest filings for that company
    subs = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    recent = subs["filings"]["recent"]
    pick = None
    for form, acc, doc, filed in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form == args.form:
            pick = (acc, doc, filed)
            break
    if pick is None:
        sys.exit(f"no {args.form} found for {ticker}")
    acc, doc, filed = pick
    print(f"latest {args.form}: filed {filed}, accession {acc}")

    if filed < args.min_filed:
        sys.exit(
            f"REFUSING: filed {filed} is before {args.min_filed}. "
            "The demo needs a post-cutoff filing. Override with --min-filed if you are sure."
        )

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
    html_path = OUT / f"{ticker}_{args.form}_{filed}.html"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        html_path.write_bytes(r.read())
    print(f"downloaded -> {html_path.relative_to(ROOT)} ({html_path.stat().st_size/1e6:.1f} MB)")

    # parse with Docling
    from docling.document_converter import DocumentConverter

    md = DocumentConverter().convert(str(html_path)).document.export_to_markdown()
    md_path = html_path.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    print(f"parsed     -> {md_path.relative_to(ROOT)} ({len(md):,} chars)")
    print("\nnext: build data/facts/fact_sheet.csv from this file (see the template there)")


if __name__ == "__main__":
    main()
