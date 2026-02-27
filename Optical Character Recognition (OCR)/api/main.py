"""
main.py
=======
Script 7/7 - FastAPI web service for document inference

What this script does:
- Wraps the entire inference pipeline into a REST API
- Exposes a single endpoint: POST /predict
- Accepts a document image uploaded by the user
- Returns structured JSON with extracted fields

Key concepts:
- FastAPI: A Python web framework for building APIs quickly.
- REST API (Representational State Transfer API): A standard way for
  systems to communicate over the internet using HTTP requests.
- Endpoint: A URL that accepts requests. Ours is POST /predict.
- Uvicorn: The server that runs our FastAPI app.
"""

import io
import sys
import os
import json
import torch
import easyocr
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    BEST_MODEL_DIR,
    PRETRAINED_MODEL,
    MAX_TOKEN_LENGTH,
    ID2LABEL,
    API_HOST,
    API_PORT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OCR Document Extractor",
    description="Upload a document image and get structured fields extracted using LayoutLMv3.",
    version="1.0.0",
)

# Load everything once at startup — not on every request
logger.info("Loading LayoutLMv3 processor...")
processor = LayoutLMv3Processor.from_pretrained(PRETRAINED_MODEL, apply_ocr=False)

logger.info(f"Loading model from: {BEST_MODEL_DIR}")
model = LayoutLMv3ForTokenClassification.from_pretrained(str(BEST_MODEL_DIR))
model.eval()

logger.info("Loading EasyOCR reader...")
ocr_reader = easyocr.Reader(["en"], gpu=False)

logger.info("All models loaded. API is ready.")


# ── Helper functions ───────────────────────────────────────────────────────────

def normalize_bbox(bbox, width, height):
    """Normalize bounding box to [0, 1000] range for LayoutLMv3."""
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def easyocr_to_words_and_boxes(ocr_results, image_width, image_height):
    """Convert EasyOCR output to flat lists of words and normalized bounding boxes."""
    words = []
    boxes = []

    for (polygon, text, confidence) in ocr_results:
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        words.append(text)
        boxes.append(normalize_bbox([x_min, y_min, x_max, y_max],
                                    image_width, image_height))

    return words, boxes


def build_structured_output(word_label_pairs):
    """Group consecutive labeled words into structured fields."""
    result = {"headers": [], "questions": [], "answers": [], "other": []}

    current_entity = []
    current_type   = None

    for word, label in word_label_pairs:

        if label.startswith("B-"):
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                if current_type == "HEADER":
                    result["headers"].append(entity_text)
                elif current_type == "QUESTION":
                    result["questions"].append(entity_text)
                elif current_type == "ANSWER":
                    result["answers"].append(entity_text)
                else:
                    result["other"].append(entity_text)

            current_entity = [word]
            current_type   = label[2:]

        elif label.startswith("I-") and current_type:
            current_entity.append(word)

        else:
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                if current_type == "HEADER":
                    result["headers"].append(entity_text)
                elif current_type == "QUESTION":
                    result["questions"].append(entity_text)
                elif current_type == "ANSWER":
                    result["answers"].append(entity_text)
                else:
                    result["other"].append(entity_text)
            current_entity = []
            current_type   = None

    if current_entity and current_type:
        entity_text = " ".join(current_entity)
        if current_type == "HEADER":
            result["headers"].append(entity_text)
        elif current_type == "QUESTION":
            result["questions"].append(entity_text)
        elif current_type == "ANSWER":
            result["answers"].append(entity_text)

    return result


def run_pipeline(image: Image.Image) -> dict:
    """
    Full inference pipeline on a PIL image.
    Same logic as inference.py but packaged as a function for the API.
    """
    img_width, img_height = image.size

    # OCR
    image_np    = np.array(image)
    ocr_results = ocr_reader.readtext(image_np)

    if not ocr_results:
        raise HTTPException(status_code=422, detail="No text detected in the image.")

    words, boxes = easyocr_to_words_and_boxes(ocr_results, img_width, img_height)

    # Tokenize
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKEN_LENGTH,
        return_tensors="pt",
    )

    # Model inference
    with torch.no_grad():
        outputs = model(**encoding)

    predicted_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
    token_ids     = encoding["input_ids"].squeeze(0).tolist()

    # Map token labels back to word labels
    word_label_pairs = []
    word_idx  = 0
    prev_text = None

    for token_id, label_id in zip(token_ids, predicted_ids):
        if token_id in processor.tokenizer.all_special_ids:
            continue
        if label_id == -100:
            continue

        token_text = processor.tokenizer.decode([token_id]).strip()
        label_name = ID2LABEL.get(label_id, "O")

        if word_idx < len(words) and token_text != prev_text:
            word_label_pairs.append((words[word_idx], label_name))
            word_idx  += 1
            prev_text  = token_text

    structured = build_structured_output(word_label_pairs)
    structured["metadata"] = {
        "words_detected": len(ocr_results),
        "image_size": f"{img_width}x{img_height}",
    }

    return structured


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — visit http://localhost:8000/ to confirm API is running."""
    return {"status": "running", "message": "OCR Document Extractor API is live."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts a document image and returns extracted fields.

    How to call:
      curl -X POST "http://localhost:8000/predict" -F "file=@your_document.png"

    Or via Python:
      import requests
      response = requests.post(
          "http://localhost:8000/predict",
          files={"file": open("your_document.png", "rb")}
      )
      print(response.json())
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload JPEG, PNG or TIFF."
        )

    logger.info(f"[REQUEST] Received: {file.filename}  type: {file.content_type}")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info(f"[REQUEST] Image size: {image.size}")

    result = run_pipeline(image)

    logger.info(f"[REQUEST] Done. Words detected: {result['metadata']['words_detected']}")
    logger.info(f"[REQUEST] Result:\n{json.dumps(result, indent=2)}")

    return JSONResponse(content=result)


# ── Run server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info("Visit http://localhost:8000/docs for interactive documentation")
    uvicorn.run(app, host=API_HOST, port=API_PORT)