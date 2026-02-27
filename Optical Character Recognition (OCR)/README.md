# OCR Invoice Project

An end-to-end document understanding system using **LayoutLMv3** — a multimodal transformer that combines text, layout (bounding boxes), and image information to extract structured data from scanned documents.

The system takes a document image as input, uses **Optical Character Recognition (OCR)** to read the text, and classifies each word as a **question**, **answer**, **header**, or **other** — returning a clean structured JSON output.

---

## What is OCR?

**OCR (Optical Character Recognition)** is the technology that reads and extracts text from images or scanned documents, converting visual text into machine-readable characters. In this project, OCR is performed by **EasyOCR**, which detects words and their positions (bounding boxes) on the page.

---

## Project Structure

```
Optical Character Recognition (OCR)/
│
├── api/
│   └── main.py                  # FastAPI web service — POST /predict endpoint
│
├── data/
│   ├── raw/funsd_subset/        # Downloaded FUNSD dataset (created by data_loader.py)
│   ├── processed/funsd_layoutlm/ # Tokenized tensors ready for training
│   ├── plots/                   # Training and evaluation plots
│   └── inference_output/        # JSON output and annotated images from inference
│
├── models/
│   ├── best_model/              # Best checkpoint saved during training
│   └── final_model/             # Final checkpoint after all epochs
│
├── src/
│   ├── data_loader.py           # Script 1 — Download and subset FUNSD dataset
│   ├── preprocessing.py         # Script 2 — Tokenize documents for LayoutLMv3
│   ├── train_layoutlm.py        # Script 3 — Fine-tune the model
│   ├── evaluate.py              # Script 4 — Evaluate model on test set
│   └── inference.py             # Script 5 — Run inference on a new document image
│
├── utils/
│   ├── config.py                # Central configuration (paths, hyperparameters, labels)
│   └── logger.py                # Centralized logging setup
│
├── Dockerfile                   # Docker container definition
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Pipeline Overview

The project has two separate phases:

### Phase 1 — Training (run once)

```
data_loader.py  →  preprocessing.py  →  train_layoutlm.py  →  evaluate.py
     ↓                   ↓                      ↓                   ↓
Download FUNSD     Tokenize docs          Fine-tune model      Measure accuracy
  dataset          to tensors            save to disk          on test set
```

### Phase 2 — Production (runs every time)

```
New document image
        ↓
inference.py  (EasyOCR → LayoutLMv3 → structured JSON)
        ↓
api/main.py   (wraps inference in HTTP endpoint)
```

---

## Dataset

**FUNSD (Form Understanding in Noisy Scanned Documents)**  
A public dataset of real scanned forms where every word is labeled as `question`, `answer`, `header`, or `other`. Used to fine-tune the model on document understanding tasks.

- Source: [HuggingFace — nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd)
- Training subset used: 30 samples (CPU-friendly demo)
- Test subset used: 10 samples

---

## Model

**LayoutLMv3 (Layout Language Model v3)** by Microsoft  
A multimodal transformer pre-trained on millions of documents. It understands:

| Modality | What it provides |
|---|---|
| Text | The words and their meaning |
| Layout (bounding boxes) | Where each word is positioned on the page |
| Image | The visual appearance of the document |

We fine-tune it for **NER (Named Entity Recognition)** using **BIO tagging**:

| Label | Meaning | Example |
|---|---|---|
| `B-QUESTION` | Start of a field label | `"Name:"` |
| `I-QUESTION` | Continuation of field label | `"First Name:"` |
| `B-ANSWER` | Start of a value | `"John"` |
| `I-ANSWER` | Continuation of a value | `"John Smith"` |
| `B-HEADER` | Start of a section title | `"Personal Info"` |
| `I-HEADER` | Continuation of title | `"Personal Information"` |
| `O` | Not part of any entity | page numbers, noise |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline

```bash
# Step 1 — Download dataset
python src/data_loader.py --subset_size 30

# Step 2 — Preprocess
python src/preprocessing.py

# Step 3 — Train
python src/train_layoutlm.py --epochs 5 --batch_size 2

# Step 4 — Evaluate
python src/evaluate.py
```

All paths and default parameters are defined in `utils/config.py`.  
Any argument can be overridden via the command line:

```bash
python src/train_layoutlm.py --epochs 10 --learning_rate 3e-5 --batch_size 4
```

---

## Inference

### Command line

```bash
python src/inference.py --image_path path/to/document.jpg
```

Output is saved to `data/inference_output/`:
- `extracted_fields.json` — structured JSON with questions, answers, headers
- `annotated_document.png` — original image with colored bounding boxes per label

### Extract a test image from the dataset

```bash
python save_test_image.py
python src/inference.py --image_path test_doc.png
```

### Example output

```json
{
    "headers": ["MEMORANDUM"],
    "questions": ["Date:", "To:", "From:", "Subject:"],
    "answers": ["January 12 2024", "John Smith", "Jane Doe", "Q1 Report"],
    "other": [],
    "metadata": {
        "words_detected": 47,
        "image_size": "762x1000"
    }
}
```

---

## API

Start the web service:

```bash
python api/main.py
```

The API runs at `http://localhost:8000`.  
Visit `http://localhost:8000/docs` for the interactive documentation UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Upload image, get structured JSON |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@document.png"
```

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("document.png", "rb")}
)
print(response.json())
```

---

## Docker

```bash
# Build the image
docker build -t ocr-invoice-app .

# Run the container
docker run -p 8000:8000 ocr-invoice-app
```

The API will be available at `http://localhost:8000`.

---

## Evaluation Results

Results on 10-sample test subset (CPU training, 5 epochs, 30 training samples):

| Label | Precision | Recall | F1 |
|---|---|---|---|
| O | 0.444 | 0.421 | 0.432 |
| B-HEADER | 0.857 | 0.222 | 0.353 |
| I-HEADER | 0.355 | 0.423 | 0.386 |
| B-QUESTION | 0.753 | 0.711 | 0.731 |
| I-QUESTION | 0.589 | 0.651 | 0.619 |
| B-ANSWER | 0.703 | 0.812 | 0.754 |
| I-ANSWER | 0.791 | 0.764 | 0.777 |
| **Weighted F1** | | | **0.674** |

> These results are expected for a 30-sample demo subset. Training on the full FUNSD dataset (149 documents) would significantly improve performance.

---

## Tech Stack

| Component | Technology |
|---|---|
| Document understanding model | LayoutLMv3 (Microsoft) |
| OCR engine | EasyOCR |
| Deep learning framework | PyTorch |
| Model library | HuggingFace Transformers |
| Dataset library | HuggingFace Datasets |
| API framework | FastAPI |
| API server | Uvicorn |
| Containerization | Docker |
| Evaluation metrics | scikit-learn |
| Plotting | matplotlib, seaborn |

---

## Key Concepts

- **OCR (Optical Character Recognition)**: Extracting text from images
- **NER (Named Entity Recognition)**: Classifying words into predefined categories
- **BIO Tagging**: B = Beginning of entity, I = Inside entity, O = Outside
- **Fine-tuning**: Adapting a pretrained model to a specific task with additional training
- **Multimodal model**: A model that processes multiple types of input (text + layout + image)
- **REST API (Representational State Transfer API)**: Standard for web service communication

---
