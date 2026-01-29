FROM python:3.11-slim

# Install FFmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Environment variables
ENV HF_TOKEN=${HF_TOKEN}
ENV API_KEY=${API_KEY}
ENV MODEL_REPO=nvidia/personaplex-7b-v1
ENV DEVICE=cuda
ENV MAX_AUDIO_SIZE=10485760

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
