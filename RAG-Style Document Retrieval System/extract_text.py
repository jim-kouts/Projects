# extract_text.py
# ------------------------------------------------------------
# Extract text from PDF files and store them in a dataframe.
#
# Input:
#   data/raw_pdfs/*.pdf
#
# Output:
#   data/processed/docs.parquet
#   data/processed/docs_preview.csv
#
# Notes:
# - We keep it simple: one row per PDF.
# - We also store per-page text in a list (page_texts) so later we can
#   add page-level citations if we want.
# - We print dataframe previews frequently to see what changed.
# ------------------------------------------------------------

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def clean_text(s: str) -> str:
    """Basic cleanup to make text nicer. Keep it minimal."""
    if s is None:
        return ""
    # Replace multiple whitespace with single spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs -> docs.parquet")
    parser.add_argument("--pdf_dir", type=str, default="data/raw_pdfs", help="Folder containing PDF files.")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Folder to save extracted outputs.")
    parser.add_argument("--max_pdfs", type=int, default=0, help="If >0, process only first N PDFs (debug).")
    parser.add_argument("--min_chars", type=int, default=200, help="Drop documents with extracted text < min_chars.")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("extract_text")

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(list(pdf_dir.glob("*.pdf")))
    if args.max_pdfs and args.max_pdfs > 0:
        pdf_paths = pdf_paths[: args.max_pdfs]

    logger.info("PDF folder: %s", pdf_dir)
    logger.info("Found %d PDFs", len(pdf_paths))

    if len(pdf_paths) == 0:
        logger.warning("No PDFs found. Put PDFs into data/raw_pdfs and rerun.")
        return

    # ------------------------------------------------------------
    # Extract text from each PDF
    # ------------------------------------------------------------
    rows = []
    n_ok = 0
    n_fail = 0

    for i, pdf_path in enumerate(pdf_paths, start=1):
        try:
            reader = PdfReader(str(pdf_path))
            n_pages = len(reader.pages)

            page_texts = []
            for p in range(n_pages):
                # extract_text can sometimes return None
                t = reader.pages[p].extract_text() or ""
                t = clean_text(t)
                page_texts.append(t)

            full_text = clean_text(" ".join(page_texts))
            char_count = len(full_text)

            rows.append(
                {
                    "doc_id": f"doc_{i:05d}",          # stable-ish id for this run
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "n_pages": int(n_pages),
                    "char_count": int(char_count),
                    "text": full_text,
                    "page_texts": page_texts,         # list of strings, one per page
                }
            )
            n_ok += 1

        except Exception as e:
            n_fail += 1
            logger.warning("Failed reading %s | error=%s", pdf_path.name, str(e))

        # Print progress every 20 PDFs
        if i % 20 == 0:
            logger.info("Progress: %d/%d processed | ok=%d fail=%d", i, len(pdf_paths), n_ok, n_fail)

    # Create dataframe
    df = pd.DataFrame(rows)

    logger.info("Raw extracted dataframe shape: %s", df.shape)

    # Print a quick preview
    if len(df) > 0:
        print("\nPreview (before filtering):")
        print(df[["doc_id", "file_name", "n_pages", "char_count"]].head(10))

    # ------------------------------------------------------------
    # Filter out documents with too little text
    # (Some PDFs may be scanned images -> extraction gives near-empty text)
    # ------------------------------------------------------------
    before = len(df)
    df = df[df["char_count"] >= args.min_chars].copy()
    after = len(df)

    logger.info("Filtered docs by min_chars=%d | kept %d/%d", args.min_chars, after, before)

    # Print a preview again (after filtering)
    if len(df) > 0:
        print("\nPreview (after filtering):")
        print(df[["doc_id", "file_name", "n_pages", "char_count"]].head(10))

        # Show the smallest and largest char_count docs (quick sanity check)
        print("\nSmallest docs by char_count:")
        print(df.sort_values("char_count").head(5)[["file_name", "n_pages", "char_count"]])

        print("\nLargest docs by char_count:")
        print(df.sort_values("char_count", ascending=False).head(5)[["file_name", "n_pages", "char_count"]])

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    out_parquet = out_dir / "docs.parquet"
    out_csv = out_dir / "docs_preview.csv"

    df.to_parquet(out_parquet, index=False)
    # Save a small CSV preview (without full text, so it stays small)
    df[["doc_id", "file_name", "n_pages", "char_count"]].to_csv(out_csv, index=False)

    logger.info("Saved docs parquet: %s", out_parquet)
    logger.info("Saved docs preview: %s", out_csv)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
