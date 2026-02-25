# build_index.py
# ------------------------------------------------------------
# Build a vector search index for text chunks.
#
# Input:
#   data/processed/chunks.parquet
#
# Output:
#   indexes/faiss.index
#   indexes/chunks_meta.parquet
#   reports/index_build_report.json
#
# We use:
# - sentence-transformers for embeddings
# - FAISS for fast nearest-neighbor search
#
# Prints:
# - chunks dataframe preview
# - embedding matrix shape
# - a few example metadata rows
# ------------------------------------------------------------

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.parquet")
    parser.add_argument("--chunks_path", type=str, default="data/processed/chunks.parquet", help="Input chunks parquet.")
    parser.add_argument("--index_dir", type=str, default="indexes", help="Folder to save FAISS index + metadata.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Folder to save build report.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name (small + strong).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max_chunks", type=int, default=0, help="If >0, index only first N chunks (debug).")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended for cosine).")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("build_index")

    chunks_path = Path(args.chunks_path)
    index_dir = Path(args.index_dir)
    reports_dir = Path(args.reports_dir)

    index_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {chunks_path}")

    # ------------------------------------------------------------
    # Load chunks
    # ------------------------------------------------------------
    df = pd.read_parquet(chunks_path)
    logger.info("Loaded chunks: %s", df.shape)

    print("\nChunks preview:")
    print(df[["chunk_id", "doc_id", "chunk_index", "n_words", "file_name"]].head(10))

    if args.max_chunks and args.max_chunks > 0:
        df = df.head(args.max_chunks).copy()
        logger.info("Using only first %d chunks for debug: %s", args.max_chunks, df.shape)

    # Keep only metadata we need aligned with the FAISS rows
    meta_cols = ["chunk_id", "doc_id", "file_name", "chunk_index", "n_words", "n_chars"]
    df_meta = df[meta_cols].copy()

    # ------------------------------------------------------------
    # Load embedding model
    # ------------------------------------------------------------
    logger.info("Loading embedding model: %s", args.model_name)
    model = SentenceTransformer(args.model_name)

    # ------------------------------------------------------------
    # Embed all chunks
    # ------------------------------------------------------------
    texts = df["chunk_text"].astype(str).tolist()

    t0 = time.time()
    logger.info("Embedding %d chunks ...", len(texts))

    # encode() returns float32 np array by default
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we handle normalization ourselves if needed
    )

    emb = emb.astype(np.float32)
    t1 = time.time()

    logger.info("Embeddings shape: %s", emb.shape)
    logger.info("Embedding time: %.2f sec", t1 - t0)

    # Optional L2 normalization (useful when you want cosine similarity)
    # For cosine similarity with FAISS IndexFlatIP, normalize and use inner product.
    if args.normalize:
        logger.info("Applying L2 normalization to embeddings (cosine similarity setup).")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        emb = emb / norms

    # Print a couple rows so you see it’s not empty
    print("\nEmbedding preview (first row, first 8 dims):")
    print(emb[0, :8])

    # ------------------------------------------------------------
    # Build FAISS index
    # Choose index type:
    # - If normalize=True -> use inner product as cosine similarity (IndexFlatIP)
    # - else -> use L2 distance (IndexFlatL2)
    # ------------------------------------------------------------
    dim = emb.shape[1]

    if args.normalize:
        index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
        metric_name = "cosine (via inner product on normalized vectors)"
    else:
        index = faiss.IndexFlatL2(dim)
        metric_name = "L2 distance"

    logger.info("Building FAISS index: %s", metric_name)
    index.add(emb)  # add all vectors

    logger.info("FAISS index size (ntotal): %d", index.ntotal)

    # ------------------------------------------------------------
    # Save index + metadata
    # ------------------------------------------------------------
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks_meta.parquet"
    report_path = reports_dir / "index_build_report.json"

    faiss.write_index(index, str(index_path))
    df_meta.to_parquet(meta_path, index=False)

    logger.info("Saved FAISS index: %s", index_path)
    logger.info("Saved metadata:   %s", meta_path)

    # Build report JSON (simple)
    report = {
        "chunks_path": str(chunks_path),
        "n_chunks": int(len(df)),
        "embedding_model": args.model_name,
        "embedding_dim": int(dim),
        "batch_size": int(args.batch_size),
        "normalize": bool(args.normalize),
        "faiss_metric": metric_name,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "embedding_time_sec": float(t1 - t0),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved report: %s", report_path)

    # Print metadata preview so you can see what will be used for citations
    print("\nMetadata preview (first 10):")
    print(df_meta.head(10))

    logger.info("Done ✅")
    logger.info("Next step: run query_rag.py to ask questions and retrieve relevant chunks.")


if __name__ == "__main__":
    main()
