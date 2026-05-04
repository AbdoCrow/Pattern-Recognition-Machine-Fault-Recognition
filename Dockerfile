# =============================================================================
# Dockerfile — Containerised inference environment for submission
# =============================================================================
# OWNER: Osama
#
# HOW THE EXAMINER USES THIS:
#   1. They build the image:    docker build -t machine-classifier .
#   2. They mount the test data: docker run -v /path/to/test/data:/app/data machine-classifier
#   3. Your infer.py runs automatically and produces results.txt + time.txt
#
# CRITICAL: The examiner places WAV files in /app/data inside the container.
#           Your code must read from that exact path.
# =============================================================================

FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .

# Install dependencies — no cache to keep image small
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
# Install Python dependencies 
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Copy the entire project into the container
COPY . .

# Create the data directory (examiner will mount over this)
RUN mkdir -p /app/data

# Create checkpoints directory (model weights should be included in the image)
RUN mkdir -p /app/checkpoints

# Default command: run inference
CMD ["python", "infer.py"]
