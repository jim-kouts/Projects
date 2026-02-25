# chunk_text.py
# ------------------------------------------------------------
# Convert docs.parquet (one row per PDF) into chunks.parquet (one row per chunk).
#
# Input:
#   data/processed/docs.parquet
#
# Output:
#   data/processed/chunks.parquet
#   reports/chunk_stats.png
#
# Chunking method (simple word-based):
# - split text into words
# - take chunks of size chunk_words with overlap overlap_words
#
# We print stats frequently so you can SEE how the dataset changes.
# ------------------------------------------------------------

import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted PDF text -> chunks.parquet")
    parser.add_argument("--docs_path", type=str, default="data/processed/docs.parquet", help="Input docs parquet.")
    parser.add_argument("--out_path", type=str, default="data/processed/chunks.parquet", help="Output chunks parquet.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Where to save plots.")
    parser.add_argument("--chunk_words", type=int, default=800, help="Words per chunk.")
    parser.add_argument("--overlap_words", type=int, default=100, help="Overlap words between chunks.")
    parser.add_argument("--min_chunk_words", type=int, default=50, help="Drop chunks smaller than this.")
    parser.add_argument("--max_docs", type=int, default=0, help="If >0, only chunk first N docs (debug).")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("chunk_text")

    docs_path = Path(args.docs_path)
    out_path = Path(args.out_path)
    reports_dir = Path(args.reports_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs parquet: {docs_path}")

    # ------------------------------------------------------------
    # Load docs
    # ------------------------------------------------------------
    df_docs = pd.read_parquet(docs_path)

    logger.info("Loaded docs: %s", df_docs.shape)
    print("\nDocs preview:")
    print(df_docs[["doc_id", "file_name", "n_pages", "char_count"]].head(5))

    # Optional debug: limit number of docs
    if args.max_docs and args.max_docs > 0:
        df_docs = df_docs.head(args.max_docs).copy()
        logger.info("Using only first %d docs for debug: %s", args.max_docs, df_docs.shape)

    # Sanity on parameters
    if args.overlap_words >= args.chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words.")

    # ------------------------------------------------------------
    # Chunk each doc
    # ------------------------------------------------------------
    rows = []
    total_docs = len(df_docs)
    global_chunk_id = 0

    for i, row in enumerate(df_docs.itertuples(index=False), start=1):
        doc_id = row.doc_id
        file_name = row.file_name
        text = row.text or ""

        # Convert to word list (simple split)
        words = text.split()
        n_words_doc = len(words)

        # If doc is empty, skip
        if n_words_doc == 0:
            continue

        step = args.chunk_words - args.overlap_words
        chunk_index = 0

        # Generate chunks
        for start in range(0, n_words_doc, step):
            end = min(start + args.chunk_words, n_words_doc)
            chunk_words_list = words[start:end]
            n_words = len(chunk_words_list)

            # Skip tiny chunks
            if n_words < args.min_chunk_words:
                continue

            chunk_text = " ".join(chunk_words_list)
            global_chunk_id += 1

            rows.append(
                {
                    "chunk_id": f"chunk_{global_chunk_id:07d}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "chunk_index": int(chunk_index),
                    "chunk_text": chunk_text,
                    "n_words": int(n_words),
                    "n_chars": int(len(chunk_text)),
                }
            )
            chunk_index += 1

        # Print progress every 10 docs
        if i % 10 == 0:
            logger.info("Chunked %d/%d docs | chunks so far: %d", i, total_docs, len(rows))

            # Show a tiny preview of what chunks look like
            if len(rows) > 0:
                df_tmp = pd.DataFrame(rows[-3:])
                print("\nLast 3 chunks preview:")
                print(df_tmp[["chunk_id", "doc_id", "chunk_index", "n_words", "n_chars", "file_name"]])

    # ------------------------------------------------------------
    # Build chunks dataframe
    # ------------------------------------------------------------
    df_chunks = pd.DataFrame(rows)
    logger.info("Chunks dataframe shape: %s", df_chunks.shape)

    # Print dataset stats frequently
    if len(df_chunks) > 0:
        print("\nChunks preview (head):")
        print(df_chunks[["chunk_id", "doc_id", "chunk_index", "n_words", "n_chars", "file_name"]].head(10))

        print("\nChunk length stats (words):")
        print(df_chunks["n_words"].describe())

        print("\nChunk length stats (chars):")
        print(df_chunks["n_chars"].describe())

        # Show top docs by number of chunks (helps spot huge PDFs)
        counts = df_chunks.groupby("file_name")["chunk_id"].count().sort_values(ascending=False).head(10)
        print("\nTop-10 docs by number of chunks:")
        print(counts)

    # ------------------------------------------------------------
    # Save chunks parquet
    # ------------------------------------------------------------
    df_chunks.to_parquet(out_path, index=False)
    logger.info("Saved chunks parquet: %s", out_path)

    # ------------------------------------------------------------
    # Plot: histogram of chunk sizes (words)
    # ------------------------------------------------------------
    if len(df_chunks) > 0:
        plt.figure()
        plt.hist(df_chunks["n_words"], bins=30)
        plt.xlabel("Chunk size (words)")
        plt.ylabel("Count")
        plt.title("Chunk size distribution")
        plot_path = reports_dir / "chunk_stats.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Saved plot: %s", plot_path)

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
