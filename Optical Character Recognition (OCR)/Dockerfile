# Dockerfile
# ==========
# Packages the entire OCR Document Extractor into a Docker container.
#
# What is Docker?
# Docker creates a "container" — an isolated environment that has everything
# the app needs to run: Python, all libraries, the model, the code.
# This means the app runs identically on any machine, server or cloud.
#
# Key concepts:
# - Image: The blueprint of the container (built from this file)
# - Container: A running instance of the image
# - Layer: Each instruction in this file creates a layer.
#   Docker caches layers so rebuilding is fast if only code changes.
#
# How to build and run:
#   docker build -t ocr-invoice-app .
#   docker run -p 8000:8000 ocr-invoice-app

# ── Base image ─────────────────────────────────────────────────────────────────
# We start from an official Python 3.11 image (slim = smaller size, no extras)
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# These are OS-level packages needed by OpenCV and EasyOCR internally
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
# All commands from here on run inside /app inside the container
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
# We copy requirements.txt first and install before copying the rest of the code.
# Why? Docker caches this layer — if only your code changes but not requirements,
# it skips the slow pip install step on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY api/      ./api/
COPY src/      ./src/
COPY models/   ./models/
COPY utils/    ./utils/

# ── Expose port ────────────────────────────────────────────────────────────────
# Tell Docker that the container listens on port 8000
# This is the same port FastAPI runs on
EXPOSE 8000

# ── Start command ──────────────────────────────────────────────────────────────
# This runs when the container starts.
# We start the FastAPI server directly with uvicorn.
CMD ["python", "api/main.py"]