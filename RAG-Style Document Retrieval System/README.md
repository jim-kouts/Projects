### RAG-Style Document Retrieval System (No LLM Required)

This project implements a full Retrieval-Augmented Generation (RAG) backend pipeline using scientific PDFs (e.g. arXiv papers). It focuses purely on high-quality document retrieval and evaluation, without relying on a large language model for generation.

# The system:

1) Downloads and samples PDFs from arXiv

2) Extracts and cleans text

3) Splits documents into overlapping chunks

4) Generates vector embeddings

5) Builds a FAISS vector index

6) Retrieves top-K relevant chunks for a given question

7) Evaluates retrieval performance using hit@K metrics

8) Saves structured reports and diagnostic artifacts


# This project demonstrates:

1) End-to-end data pipeline construction

2) Vector search using FAISS

3) Embedding model integration

4) Retrieval evaluation methodology

5) Failure analysis for system debugging

6) Production-style artifact saving


# Workflow:

python .\arxiv_sample_and_download_pdfs.py --n_samples 300
python .\extract_text.py
python .\chunk_text.py
python .\build_index.py --normalize
python .\query_rag.py --question "Black holes and cosmos" --top_k 5 --normalize
python .\make_qa_set.py --n_questions 
python .\evaluate_retrieval.py --normalize 