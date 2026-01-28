FROM python:3.11-slim

# Make Python behavior nicer inside containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# (Optional) install minimal OS tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Work directory inside the container
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . /app

# Default command (can be overridden in `docker run ...`)
CMD ["python", "src/train.py", "--img_size", "64", "--epochs", "2"]
