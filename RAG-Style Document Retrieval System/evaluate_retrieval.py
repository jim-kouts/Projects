# evaluate_retrieval.py
# ------------------------------------------------------------
# Evaluate retrieval quality WITHOUT using an LLM.
#
# For each question in the QA set:
# 1) embed question
# 2) retrieve top-k chunks via FAISS
# 3) check if correct PDF is present in the retrieved results
#
# Outputs:
#   reports/retrieval_eval.json
#   reports/retrieval_hit_at_k.png
#   reports/retrieval_failures.parquet
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import faiss
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit-rate / recall@k (no LLM).")
    parser.add_argument("--qa_path", type=str, default="data/qa/qa_set.parquet", help="QA set parquet.")
    parser.add_argument("--index_path", type=str, default="indexes/faiss.index", help="FAISS index file.")
    parser.add_argument("--meta_path", type=str, default="indexes/chunks_meta.parquet", help="Chunk metadata parquet.")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model (must match indexing).")
    parser.add_argument("--normalize", action="store_true", help="Use if you built index with normalized embeddings.")
    parser.add_argument("--k_values", type=str, default="1,3,5,10", help="Comma-separated K values to evaluate.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Where to save reports.")
    parser.add_argument("--max_questions", type=int, default=0, help="If >0, evaluate only first N questions (debug).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("eval_retrieval")

    qa_path = Path(args.qa_path)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not qa_path.exists():
        raise FileNotFoundError(f"Missing: {qa_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    # Parse K values
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    k_values = sorted(list(set([k for k in k_values if k > 0])))
    if not k_values:
        raise ValueError("No valid k_values provided.")

    k_max = max(k_values)
    logger.info("Evaluating K values: %s (max=%d)", k_values, k_max)

    # Load QA set
    df_qa = pd.read_parquet(qa_path)
    logger.info("Loaded QA set: %s", df_qa.shape)
    print("\nQA preview:")
    print(df_qa[["question", "pdf_name"]].head(5))

    if args.max_questions and args.max_questions > 0:
        df_qa = df_qa.head(args.max_questions).copy()
        logger.info("Using only first %d questions for debug: %s", args.max_questions, df_qa.shape)

    # Load index + metadata
    logger.info("Loading FAISS index: %s", index_path)
    index = faiss.read_index(str(index_path))

    logger.info("Loading metadata: %s", meta_path)
    df_meta = pd.read_parquet(meta_path)
    logger.info("Meta shape: %s", df_meta.shape)

    # Load embedding model
    logger.info("Loading embedding model: %s", args.embed_model)
    model = SentenceTransformer(args.embed_model)

    # Prepare counters
    hits = {k: 0 for k in k_values}
    failures = []

    # Evaluate each question
    for i, r in enumerate(df_qa.itertuples(index=False), start=1):
        question = str(getattr(r, "question"))
        true_pdf = str(getattr(r, "pdf_name"))

        # Embed question
        q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        if args.normalize:
            norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-12)
            q_emb = q_emb / norm

        # Retrieve top k_max rows from FAISS
        distances, indices = index.search(q_emb, k_max)
        idxs = indices[0].tolist()
        scores = distances[0].tolist()

        # Map retrieved rows to pdf names
        retrieved_pdfs = []
        retrieved_rows = []
        for rank, (row_idx, score) in enumerate(zip(idxs, scores), start=1):
            if row_idx < 0 or row_idx >= len(df_meta):
                continue
            m = df_meta.iloc[row_idx]
            pdf_name = str(m["file_name"])
            retrieved_pdfs.append(pdf_name)
            retrieved_rows.append(
                {"rank": rank, "score": float(score), "pdf_name": pdf_name, "chunk_id": str(m["chunk_id"]), "chunk_index": int(m["chunk_index"])}
            )

        # For each K, check if true pdf appears in top-K results
        for k in k_values:
            topk = retrieved_pdfs[:k]
            if true_pdf in topk:
                hits[k] += 1

        # Save failures for analysis (use k=5 as default view)
        k_fail_view = 5 if 5 in k_values else k_values[0]
        if true_pdf not in retrieved_pdfs[:k_fail_view]:
            failures.append(
                {
                    "question": question,
                    "true_pdf": true_pdf,
                    "top_retrieved_pdfs": retrieved_pdfs[:k_fail_view],
                    "top_retrieved_rows": retrieved_rows[:k_fail_view],
                }
            )

        # Print progress every 20 questions
        if i % 20 == 0:
            logger.info("Progress: %d/%d questions evaluated", i, len(df_qa))
            # print current hit@k so you can see it live
            for k in k_values:
                logger.info("Current hit@%d = %.3f", k, hits[k] / i)

    # Compute hit rates
    n_q = len(df_qa)
    hit_rates = {f"hit@{k}": float(hits[k] / n_q) for k in k_values}

    logger.info("Final results:")
    for k in k_values:
        logger.info("hit@%d = %.3f", k, hit_rates[f"hit@{k}"])

    # Save JSON report
    report = {
        "n_questions": int(n_q),
        "k_values": k_values,
        "hit_rates": hit_rates,
        "normalize": bool(args.normalize),
        "embed_model": args.embed_model,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "qa_path": str(qa_path),
        "n_failures_saved": int(len(failures)),
    }

    json_path = reports_dir / "retrieval_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved: %s", json_path)

    # Save failures parquet (for debugging)
    df_fail = pd.DataFrame(failures)
    fail_path = reports_dir / "retrieval_failures.parquet"
    df_fail.to_parquet(fail_path, index=False)
    logger.info("Saved: %s", fail_path)

    # Plot hit@k
    ks = k_values
    ys = [hit_rates[f"hit@{k}"] for k in ks]

    plt.figure()
    plt.plot(ks, ys, marker="o", linewidth=2)
    plt.xticks(ks)
    plt.ylim(0.0, 1.0)
    plt.xlabel("K (top retrieved results)")
    plt.ylabel("Hit rate (hit@K)")
    plt.title("Retrieval Evaluation (no LLM)")
    plot_path = reports_dir / "retrieval_hit_at_k.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", plot_path)

    # Print a few failure examples to the console
    if len(df_fail) > 0:
        print("\nFailure examples (first 3):")
        print(df_fail[["question", "true_pdf", "top_retrieved_pdfs"]].head(3))

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
