# query_rag.py
# ------------------------------------------------------------
# Query a RAG (Retrieval Augmented Generation) system:
#
# 1) Embed the user's question
# 2) Use FAISS to retrieve the top-k most similar chunks
# 3) Print retrieved sources (citations)
# 4) Optionally call an LLM to answer using ONLY those chunks
#
# Inputs:
#   indexes/faiss.index
#   indexes/chunks_meta.parquet        (aligned with FAISS row order)
#   data/processed/chunks.parquet      (contains chunk_text)
#
# Outputs:
#   prints answer + sources
#   reports/last_query.json
#
# Notes:
# - Retrieval-only mode is default (no API keys required).
# - LLM mode is optional: pass --use_llm and set OPENAI_API_KEY.
# ------------------------------------------------------------

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Query RAG: retrieve top-k chunks and optionally answer with an LLM.")
    parser.add_argument("--question", type=str, default="", help="Your question. If empty, you'll be prompted.")
    parser.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve.")
    parser.add_argument("--index_path", type=str, default="indexes/faiss.index", help="FAISS index file.")
    parser.add_argument("--meta_path", type=str, default="indexes/chunks_meta.parquet", help="Chunk metadata parquet.")
    parser.add_argument("--chunks_path", type=str, default="data/processed/chunks.parquet", help="Chunks parquet (text).")
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model (must match the one used for indexing).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If you built the index with --normalize, use --normalize here too (cosine similarity setup).",
    )
    parser.add_argument("--reports_dir", type=str, default="reports", help="Where to save last_query.json")

    # Optional LLM settings
    parser.add_argument("--use_llm", action="store_true", help="Use OpenAI LLM to generate final answer.")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--max_context_chars", type=int, default=12000, help="Max context size to send to LLM.")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("query_rag")

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Ask question interactively if not provided
    question = args.question.strip()
    if not question:
        question = input("Type your question: ").strip()
    if not question:
        raise ValueError("Empty question. Provide --question or type one interactively.")

    # ------------------------------------------------------------
    # Load index + metadata + chunks
    # ------------------------------------------------------------
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    chunks_path = Path(args.chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata parquet: {meta_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {chunks_path}")

    logger.info("Loading FAISS index: %s", index_path)
    index = faiss.read_index(str(index_path))

    logger.info("Loading metadata: %s", meta_path)
    df_meta = pd.read_parquet(meta_path)
    logger.info("Meta shape: %s", df_meta.shape)
    print("\nMeta preview:")
    print(df_meta.head(5))

    logger.info("Loading chunks: %s", chunks_path)
    df_chunks = pd.read_parquet(chunks_path)
    logger.info("Chunks shape: %s", df_chunks.shape)
    print("\nChunks preview:")
    print(df_chunks[["chunk_id", "doc_id", "chunk_index", "n_words", "file_name"]].head(5))

    # Build a quick lookup from chunk_id -> chunk_text
    # This makes retrieval robust even if ordering changes.
    logger.info("Building chunk_id -> text lookup ...")
    chunk_text_map = dict(zip(df_chunks["chunk_id"].astype(str), df_chunks["chunk_text"].astype(str)))
    logger.info("Lookup size: %d", len(chunk_text_map))

    # ------------------------------------------------------------
    # Embed the question
    # ------------------------------------------------------------
    logger.info("Loading embedding model: %s", args.embed_model)
    model = SentenceTransformer(args.embed_model)

    logger.info("Embedding question ...")
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)

    if args.normalize:
        # L2 normalize so inner product acts like cosine similarity
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        q_emb = q_emb / norm

    print("\nQuestion:")
    print(question)

    # ------------------------------------------------------------
    # Retrieve top-k from FAISS
    # ------------------------------------------------------------
    top_k = int(args.top_k)
    top_k = max(1, top_k)

    # Search returns distances (or similarity) and indices
    distances, indices = index.search(q_emb, top_k)

    # Flatten results for easier handling
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    # Build retrieved list
    retrieved = []
    for rank, (idx, score) in enumerate(zip(indices, distances), start=1):
        if idx < 0 or idx >= len(df_meta):
            continue

        m = df_meta.iloc[idx].to_dict()
        chunk_id = str(m["chunk_id"])
        text = chunk_text_map.get(chunk_id, "")

        retrieved.append(
            {
                "rank": rank,
                "faiss_row": int(idx),
                "score": float(score),
                "chunk_id": chunk_id,
                "doc_id": str(m.get("doc_id", "")),
                "file_name": str(m.get("file_name", "")),
                "chunk_index": int(m.get("chunk_index", -1)),
                "n_words": int(m.get("n_words", -1)),
                "n_chars": int(m.get("n_chars", -1)),
                "chunk_text": text,
            }
        )

    # Print retrieved sources
    print("\nTop retrieved chunks (citations):")
    for r in retrieved:
        # score meaning depends on index type:
        # - If normalize+IndexFlatIP: higher is better (cosine similarity)
        # - If L2: lower is better (distance)
        print(
            f"#{r['rank']} score={r['score']:.4f} | {r['file_name']} | chunk={r['chunk_index']} | id={r['chunk_id']}"
        )

    # Print small snippet previews so you can visually validate retrieval
    print("\nChunk text snippets (first ~300 chars each):")
    for r in retrieved:
        snippet = (r["chunk_text"][:300] + "...") if len(r["chunk_text"]) > 300 else r["chunk_text"]
        print(f"\n[{r['file_name']} | chunk {r['chunk_index']}]")
        print(snippet)

    # ------------------------------------------------------------
    # Build context string for LLM (or for printing)
    # ------------------------------------------------------------
    # We include citations per chunk in the context, so the model can cite sources.
    context_parts = []
    for r in retrieved:
        citation = f"[{r['file_name']} | chunk {r['chunk_index']}]"
        part = f"{citation}\n{r['chunk_text']}"
        context_parts.append(part)

    context = "\n\n---\n\n".join(context_parts)

    # Hard cap the context size (simple safety)
    if len(context) > args.max_context_chars:
        context = context[: args.max_context_chars] + "\n\n[TRUNCATED]\n"

    # ------------------------------------------------------------
    # Retrieval-only answer (default)
    # ------------------------------------------------------------
    answer = ""
    if not args.use_llm:
        # In retrieval-only mode, we don't "generate" an answer.
        # We just show the most relevant chunks so the user can read them.
        answer = (
            "Retrieval-only mode: showing the most relevant chunks above.\n"
            "Tip: run again with --use_llm (and set OPENAI_API_KEY) to generate a direct answer."
        )
        print("\nAnswer (retrieval-only):")
        print(answer)

    # ------------------------------------------------------------
    # LLM answer (optional)
    # ------------------------------------------------------------
    if args.use_llm:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Either set it in your environment, or run without --use_llm."
            )

        # We use the modern OpenAI python client style if available.
        # Install: pip install openai
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)

            system_msg = (
                "You are a careful assistant. Answer using ONLY the provided context.\n"
                "If the context does not contain the answer, say you don't know.\n"
                "Cite sources using the citation tags exactly as shown, e.g. [file | chunk X]."
            )

            user_msg = (
                f"QUESTION:\n{question}\n\n"
                f"CONTEXT:\n{context}\n\n"
                "Return a clear answer and include citations."
            )

            logger.info("Calling LLM: %s", args.llm_model)
            resp = client.chat.completions.create(
                model=args.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )

            answer = resp.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(
                f"Failed to call OpenAI client. Make sure 'openai' is installed and your key is valid. Error: {e}"
            )

        print("\nAnswer (LLM):")
        print(answer)

    # ------------------------------------------------------------
    # Save last query report
    # ------------------------------------------------------------
    out = {
        "question": question,
        "top_k": top_k,
        "use_llm": bool(args.use_llm),
        "embed_model": args.embed_model,
        "normalize": bool(args.normalize),
        "retrieved": [
            {
                "rank": r["rank"],
                "score": r["score"],
                "chunk_id": r["chunk_id"],
                "file_name": r["file_name"],
                "chunk_index": r["chunk_index"],
                "n_words": r["n_words"],
                "n_chars": r["n_chars"],
            }
            for r in retrieved
        ],
        "answer": answer,
    }

    out_path = reports_dir / "last_query.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info("Saved: %s", out_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
