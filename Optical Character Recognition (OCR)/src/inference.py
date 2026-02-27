"""
inference.py
============
Script 6/7 - Run inference on a single document image

What this script does:
- Loads the best trained model
- Takes a document image as input (via --image_path argument)
- Runs OCR using EasyOCR to extract words and their positions
- Passes everything through the LayoutLMv3 model
- Returns a structured JSON with extracted fields
- Saves an annotated visualization of the document

Key concepts:
- EasyOCR: A Python library that reads text from images.
  Unlike the FUNSD dataset (which had pre-labeled words),
  in production we don't have annotations — we extract words ourselves.
- Inference: Running a trained model on new, unseen data (no training here).
- Aggregation: Multiple tokens from the same word get the same label,
  so we group them back into words with a single label.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from matplotlib.lines import Line2D

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    BEST_MODEL_DIR,
    PRETRAINED_MODEL,
    INFERENCE_OUT_DIR,
    MAX_TOKEN_LENGTH,
    ID2LABEL,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Color for each label when drawing bounding boxes on the image
LABEL_COLORS = {
    "O":          "gray",
    "B-HEADER":   "red",
    "I-HEADER":   "red",
    "B-QUESTION": "blue",
    "I-QUESTION": "blue",
    "B-ANSWER":   "green",
    "I-ANSWER":   "green",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def normalize_bbox(bbox, width, height):
    """
    Normalize bounding box to [0, 1000] range (required by LayoutLMv3).
    bbox format: [x_min, y_min, x_max, y_max]
    """
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def easyocr_to_words_and_boxes(ocr_results, image_width, image_height):
    """
    Convert EasyOCR output to flat lists of words and normalized bounding boxes.
    EasyOCR returns a 4-point polygon per word — we convert to [x_min, y_min, x_max, y_max].
    """
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
    """
    Group consecutive words by their entity type into structured fields.
    B- tag = start of new entity, I- tag = continuation, O = not an entity.
    """
    result = {"headers": [], "questions": [], "answers": [], "other": []}

    current_entity = []
    current_type   = None

    for word, label in word_label_pairs:

        if label.startswith("B-"):
            # Save previous entity
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
            current_type   = label[2:]  # strip "B-"

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

    # Save last entity
    if current_entity and current_type:
        entity_text = " ".join(current_entity)
        if current_type == "HEADER":
            result["headers"].append(entity_text)
        elif current_type == "QUESTION":
            result["questions"].append(entity_text)
        elif current_type == "ANSWER":
            result["answers"].append(entity_text)

    return result


def plot_annotated_image(image, ocr_results, word_label_pairs, output_path):
    """
    Draw bounding boxes on the original image colored by label type.
    Helps visually verify that the model is labeling the right regions.
    """
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(image)

    for i, ((polygon, text, _), (word, label)) in enumerate(
        zip(ocr_results, word_label_pairs)
    ):
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        color = LABEL_COLORS.get(label, "gray")
        rect  = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x_min, y_min - 2, label, fontsize=6, color=color)

    legend_elements = [
        Line2D([0], [0], color="red",   linewidth=2, label="HEADER"),
        Line2D([0], [0], color="blue",  linewidth=2, label="QUESTION"),
        Line2D([0], [0], color="green", linewidth=2, label="ANSWER"),
        Line2D([0], [0], color="gray",  linewidth=2, label="OTHER"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Annotated image saved: {output_path}")


# ── Main inference function ────────────────────────────────────────────────────

def run_inference(args):

    # Step 1: Load image
    logger.info("=== inference.py started ===")
    logger.info(f"STEP 1: Loading image from: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    img_width, img_height = image.size
    logger.info(f"Image size: {img_width} x {img_height} px")

    # Step 2: Run OCR
    logger.info("STEP 2: Running EasyOCR on image...")
    logger.info("  (first run downloads OCR model weights — may take a moment)")
    reader      = easyocr.Reader(["en"], gpu=False)
    ocr_results = reader.readtext(args.image_path)
    logger.info(f"Words detected: {len(ocr_results)}")

    if not ocr_results:
        logger.error("No text detected in image. Exiting.")
        return

    logger.info("Preview (first 5 words):")
    for polygon, text, conf in ocr_results[:5]:
        logger.info(f"  '{text}'  confidence={conf:.2f}")

    words, boxes = easyocr_to_words_and_boxes(ocr_results, img_width, img_height)

    # Step 3: Load model and processor
    logger.info(f"STEP 3: Loading model from: {args.model_dir}")
    processor = LayoutLMv3Processor.from_pretrained(PRETRAINED_MODEL, apply_ocr=False)
    model     = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    model.eval()
    logger.info("Model and processor loaded OK.")

    # Step 4: Tokenize and run model
    logger.info("STEP 4: Running model inference...")
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKEN_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predicted_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
    token_ids     = encoding["input_ids"].squeeze(0).tolist()
    logger.info(f"Inference done. Tokens processed: {len(token_ids)}")

    # Step 5: Map token labels back to words
    logger.info("STEP 5: Mapping token labels back to words...")
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

    logger.info(f"Word-label pairs: {len(word_label_pairs)}")
    logger.info("Preview (first 10):")
    for word, label in word_label_pairs[:10]:
        logger.info(f"  '{word}' -> {label}")

    # Step 6: Build structured JSON
    logger.info("STEP 6: Building structured JSON output...")
    structured = build_structured_output(word_label_pairs)
    logger.info(f"Extracted fields:\n{json.dumps(structured, indent=4)}")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "extracted_fields.json")
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=4)
    logger.info(f"JSON saved to: {json_path}")

    # Step 7: Save annotated image
    logger.info("STEP 7: Saving annotated image...")
    annotated_path = os.path.join(args.output_dir, "annotated_document.png")
    plot_annotated_image(image, ocr_results, word_label_pairs, annotated_path)

    # Summary
    logger.info("=== INFERENCE COMPLETE ===")
    logger.info(f"Input image     : {args.image_path}")
    logger.info(f"Words detected  : {len(ocr_results)}")
    logger.info(f"Headers found   : {len(structured['headers'])}")
    logger.info(f"Questions found : {len(structured['questions'])}")
    logger.info(f"Answers found   : {len(structured['answers'])}")
    logger.info(f"JSON output     : {json_path}")
    logger.info(f"Annotated image : {annotated_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference on a document image using fine-tuned LayoutLMv3"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input document image (jpg or png)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(BEST_MODEL_DIR),
        help="Path to fine-tuned model (default: from config.py)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(INFERENCE_OUT_DIR),
        help="Where to save JSON output and annotated image (default: from config.py)"
    )

    args = parser.parse_args()
    run_inference(args)