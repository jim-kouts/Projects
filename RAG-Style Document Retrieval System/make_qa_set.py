# make_qa_set.py
# ------------------------------------------------------------
# Build a small evaluation dataset (questions + ground-truth doc).
#
# We use arXiv sample metadata (title/abstract/arxiv_id) and map it
# to the downloaded PDF filenames in data/raw_pdfs/.
#
# Output:
#   data/qa/qa_set.parquet
#
# Notes:
# - No LLM. Questions are created with simple templates from titles.
# - We print previews often so you can see what's happening.
# ------------------------------------------------------------

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip version suffix like v2 from arXiv IDs."""
    arxiv_id = (arxiv_id or "").strip()
    # remove trailing version like v2, v3
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id


def main():
    parser = argparse.ArgumentParser(description="Create QA set for retrieval evaluation (no LLM).")
    parser.add_argument("--sample_meta", type=str, default="data/processed/arxiv_sample.parquet",
                        help="Sample metadata parquet produced by 00_... script.")
    parser.add_argument("--pdf_dir", type=str, default="data/raw_pdfs", help="Folder containing downloaded PDFs.")
    parser.add_argument("--out_path", type=str, default="data/qa/qa_set.parquet", help="Output QA parquet.")
    parser.add_argument("--n_questions", type=int, default=100, help="How many questions to include.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("make_qa_set")

    sample_meta = Path(args.sample_meta)
    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not sample_meta.exists():
        raise FileNotFoundError(f"Missing: {sample_meta}")
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Missing: {pdf_dir}")

    # Load sampled metadata (from Kaggle JSON sampling)
    df = pd.read_parquet(sample_meta)
    logger.info("Loaded sample metadata: %s", df.shape)
    print("\nSample meta preview:")
    print(df[["arxiv_id", "categories", "title"]].head(5))

    # Build mapping: arxiv_id -> pdf_filename
    # Your download script names files like: "<arxiv_id>_<title>.pdf"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdf_files), pdf_dir)

    id_to_pdf = {}
    for p in pdf_files:
        # filename starts with arxiv_id
        # e.g. 0704.0101_The_birth_of_string_theory.pdf
        prefix = p.name.split("_")[0]
        id_to_pdf[normalize_arxiv_id(prefix)] = p.name

    # Create QA rows
    rows = []
    for r in df.itertuples(index=False):
        arxiv_id = normalize_arxiv_id(getattr(r, "arxiv_id", ""))
        title = (getattr(r, "title", "") or "").strip()
        abstract = (getattr(r, "abstract", "") or "").strip()
        cats = (getattr(r, "categories", "") or "").strip()

        pdf_name = id_to_pdf.get(arxiv_id, "")
        if not pdf_name:
            continue

        # Simple question templates from title (no LLM)
        q1 = f"What is this paper about: {title}?"
        q2 = f"Summarize the main idea of the paper titled: {title}"
        q3 = f"What problem does the paper '{title}' try to solve?"

        # Store multiple question styles for the same doc
        rows.append({"question": q1, "arxiv_id": arxiv_id, "pdf_name": pdf_name, "categories": cats, "title": title, "abstract": abstract})
        rows.append({"question": q2, "arxiv_id": arxiv_id, "pdf_name": pdf_name, "categories": cats, "title": title, "abstract": abstract})
        rows.append({"question": q3, "arxiv_id": arxiv_id, "pdf_name": pdf_name, "categories": cats, "title": title, "abstract": abstract})

    df_qa = pd.DataFrame(rows).drop_duplicates(subset=["question"]).reset_index(drop=True)
    logger.info("Built QA candidates: %s", df_qa.shape)

    print("\nQA preview (candidates):")
    print(df_qa[["question", "pdf_name"]].head(5))

    # Sample down to n_questions
    n = min(args.n_questions, len(df_qa))
    df_qa = df_qa.sample(n=n, random_state=args.seed).reset_index(drop=True)

    logger.info("Final QA set size: %d", n)
    print("\nQA preview (final):")
    print(df_qa[["question", "pdf_name"]].head(10))

    df_qa.to_parquet(out_path, index=False)
    logger.info("Saved QA set: %s", out_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
