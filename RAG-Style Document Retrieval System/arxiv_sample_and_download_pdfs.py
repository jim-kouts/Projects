# arxiv_sample_and_download_pdfs.py
# ------------------------------------------------------------

# You downloaded a ~5GB JSON file from Kaggle (arxiv metadata dump).
# That file is NOT PDFs, it's metadata (title/abstract/authors/categories/arxiv_id).
#
# This script:
# 1) Streams the JSON line-by-line (so we don't load 5GB in memory)
# 2) Filters by category/keyword (optional)
# 3) Samples N papers
# 4) Saves the sampled metadata to data/processed/arxiv_sample.parquet
# 5) Downloads PDFs for those arXiv IDs into data/raw_pdfs/

# ------------------------------------------------------------

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests


def safe_filename(name: str) -> str:
    """
    Make a safe filename: keep letters/numbers/._- and replace other chars with '_'
    """
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:150]  # avoid crazy-long filenames


def iter_json_lines(json_path: Path):
    """
    Stream JSON records from a file that is either:
    - JSON Lines: one JSON object per line (most common for this dataset)
    - OR a big JSON array (less common). We assume JSON Lines.
    """
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is a JSON object
            yield json.loads(line)


def extract_arxiv_id(rec: dict) -> str:
    """
    Try to extract an arXiv identifier from a record.
    Kaggle arXiv metadata typically includes something like rec["id"].
    Example: "2101.12345" or "2101.12345v2"
    """
    # Most common key:
    if "id" in rec and isinstance(rec["id"], str):
        return rec["id"].strip()

    # Fallback possibilities (rare):
    for k in ["paper_id", "arxiv_id", "identifier"]:
        if k in rec and isinstance(rec[k], str):
            return rec[k].strip()

    return ""


def main():
    parser = argparse.ArgumentParser(description="Sample arXiv papers from a huge JSON and download PDFs.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="",
        help="Path to the big arXiv metadata JSON file (jsonlines). If empty, tries to find one in data/.",
    )
    parser.add_argument("--n_samples", type=int, default=200, help="How many PDFs to download.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Optional exact category filter (e.g. cs.LG, cs.AI, stat.ML). Empty = no filter.",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="",
        help="Optional keyword filter (case-insensitive) applied to title+abstract. Empty = no filter.",
    )
    parser.add_argument(
        "--max_scan",
        type=int,
        default=300000,
        help="How many records to scan from the big JSON before stopping (for speed).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between PDF downloads (be polite to arxiv.org).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("arxiv_sample")

    random.seed(args.seed)

    # ------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------
    data_dir = Path("data")
    raw_pdfs_dir = data_dir / "raw_pdfs"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"

    # Create folders if they don't exist
    raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Find the JSON path if not provided
    # ------------------------------------------------------------
    if args.json_path:
        json_path = Path(args.json_path)
    else:
        # Try to find a large .json file in data/
        candidates = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl")) + list(data_dir.glob("**/*.json"))
        # Keep the biggest ones first (likely the 5GB file)
        candidates = sorted(candidates, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        json_path = candidates[0] if candidates else Path("")

    if not json_path or not json_path.exists():
        raise FileNotFoundError(
            "Could not find the arXiv metadata JSON. "
            "Pass it explicitly: --json_path data/<your_file>.json"
        )

    logger.info("Using JSON file: %s", json_path)
    logger.info("Raw PDFs will be saved to: %s", raw_pdfs_dir)

    # Prepare filters
    category_filter = args.category.strip()
    keyword_filter = args.keyword.strip().lower()

    logger.info("Filters: category='%s' keyword='%s'", category_filter, keyword_filter)

    # ------------------------------------------------------------
    # Stream records and collect candidate rows
    # ------------------------------------------------------------
    rows = []
    scanned = 0
    kept = 0

    for rec in iter_json_lines(json_path):
        scanned += 1
        if scanned % 50000 == 0:
            logger.info("Scanned %d records | kept so far: %d", scanned, kept)

        # Stop scanning early for speed if requested
        if scanned > args.max_scan:
            logger.info("Reached max_scan=%d. Stopping scan.", args.max_scan)
            break

        arxiv_id = extract_arxiv_id(rec)
        if not arxiv_id:
            continue

        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()

        # Categories in this dataset are often a space-separated string like:
        # "cs.LG stat.ML cs.AI"
        cats = (rec.get("categories") or "").strip()

        # ---- Apply optional category filter
        if category_filter:
            # exact category match inside the categories string
            cat_list = cats.split()
            if category_filter not in cat_list:
                continue

        # ---- Apply optional keyword filter in title+abstract
        if keyword_filter:
            blob = f"{title} {abstract}".lower()
            if keyword_filter not in blob:
                continue

        kept += 1
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "categories": cats,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            }
        )

        # For memory safety: don’t keep too many rows
        # We only need enough to sample from.
        if len(rows) >= max(args.n_samples * 20, 5000):
            # We already have plenty of candidates to sample from
            break

    if len(rows) == 0:
        raise RuntimeError("No records matched your filters. Try removing filters or increasing max_scan.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id"]).reset_index(drop=True)

    logger.info("Candidate dataframe shape: %s", df.shape)
    print(df[["arxiv_id", "categories", "title"]].head(5))

    # ------------------------------------------------------------
    # Sample N papers
    # ------------------------------------------------------------
    n = min(args.n_samples, len(df))
    df_sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    logger.info("Sampled %d papers.", n)
    print(df_sample[["arxiv_id", "categories", "title"]].head(10))

    # Save sample metadata
    sample_path = processed_dir / "arxiv_sample.parquet"
    df_sample.to_parquet(sample_path, index=False)
    logger.info("Saved sample metadata: %s", sample_path)

    # ------------------------------------------------------------
    # Download PDFs
    # ------------------------------------------------------------
    session = requests.Session()
    headers = {"User-Agent": "doc-intel-rag/1.0 (educational project)"}

    downloaded = 0
    skipped = 0
    failed = 0

    for i, row in df_sample.iterrows():
        arxiv_id = row["arxiv_id"]
        title = row["title"]
        url = row["pdf_url"]

        # Make a nice filename: arxivid_title.pdf
        fname = safe_filename(f"{arxiv_id}_{title}") + ".pdf"
        out_pdf = raw_pdfs_dir / fname

        if out_pdf.exists() and out_pdf.stat().st_size > 0:
            skipped += 1
            continue

        try:
            logger.info("[%d/%d] Downloading %s", i + 1, len(df_sample), url)
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code != 200:
                failed += 1
                logger.warning("Failed (%d): %s", r.status_code, url)
                continue

            out_pdf.write_bytes(r.content)
            downloaded += 1

            # Print a small preview frequently
            if (downloaded + skipped + failed) % 20 == 0:
                logger.info("Progress: downloaded=%d skipped=%d failed=%d", downloaded, skipped, failed)

            # Be polite to arXiv
            time.sleep(args.sleep)

        except Exception as e:
            failed += 1
            logger.warning("Exception while downloading %s: %s", url, str(e))

    logger.info("DONE: downloaded=%d skipped=%d failed=%d", downloaded, skipped, failed)
    logger.info("PDF folder: %s", raw_pdfs_dir)
    logger.info("Next step: run extract_text.py to extract text from the downloaded PDFs.")


if __name__ == "__main__":
    main()
