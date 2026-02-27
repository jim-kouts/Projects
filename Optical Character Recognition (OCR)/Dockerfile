# Dockerfile
# ==========
# Packages the entire OCR Document Extractor into a Docker container.
#
# What is Docker?
# Docker creates a "container" — an isolated environment that has everything
# the app needs to run: Python, all libraries, the code.
# This means the app runs identically on any machine, server or cloud.
#
# Key concepts:
# - Image: The blueprint of the container (built from this file)
# - Container: A running instance of the image
# - Layer: Each instruction creates a cached layer for faster rebuilds
#
# NOTE: The models/ folder is NOT copied here because:
# - Model weights are too large (500MB+)
# - They are not in the GitHub repo
# - The container downloads them from HuggingFace on first run automatically
#
# How to build and run:
#   docker build -t ocr-invoice-app .
#   docker run -p 8000:8000 ocr-invoice-app

# ── Base image ─────────────────────────────────────────────────────────────────
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
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# If only code changes, pip install is skipped on rebuild (faster).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY api/   ./api/
COPY src/   ./src/
COPY utils/ ./utils/

# NOTE: models/ is intentionally excluded.
# On first run, the API downloads the base model from HuggingFace automatically.
# If you want to include a fine-tuned model, mount it as a Docker volume:
#   docker run -p 8000:8000 -v /your/local/models:/app/models ocr-invoice-app

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ──────────────────────────────────────────────────────────────
CMD ["python", "api/main.py"]