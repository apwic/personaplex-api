# PersonaPlex API

Speech-to-speech API using nvidia/personaplex-7b-v1 model deployed on RunPod.

## Prerequisites

- Hugging Face account with access to [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
- HF_TOKEN from [Hugging Face settings](https://huggingface.co/settings/tokens)
- RunPod account

## Setup

1. Accept the model license:
   - Go to https://huggingface.co/nvidia/personaplex-7b-v1
   - Click "Agree and access repository"

2. Get your HF token from https://huggingface.co/settings/tokens

3. Generate an API key for your service

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_hf_token_here
export API_KEY=your_api_key_here

# Run locally (requires GPU)
python app.py
```

## Deploy to RunPod

### Option 1: Deploy from GitHub

1. Push this directory to a GitHub repository

2. Create a new endpoint on RunPod:
   - Go to https://runpod.io/console/serverless
   - Click "Deploy" -> "From GitHub"
   - Select your repository and branch
   - Set container image: `python:3.11-slim`
   - Override command: `bash -c "pip install -r requirements.txt && python app.py"`

3. Configure environment variables:
   - `HF_TOKEN`: Your Hugging Face token
   - `API_KEY`: Your API key
   - `MODEL_REPO`: `nvidia/personaplex-7b-v1`
   - `DEVICE`: `cuda`

4. Select GPU: RTX 4090 or A100 (24GB+ VRAM recommended)

### Option 2: Build and Deploy Custom Container

```bash
# Build image
docker build -t personaplex-api .

# Test locally
docker run -p 8000:8000 \
  -e HF_TOKEN=your_hf_token \
  -e API_KEY=your_api_key \
  personaplex-api

# Push to Docker Hub or ECR
docker tag personaplex-api yourusername/personaplex-api
docker push yourusername/personaplex-api

# Deploy to RunPod using your custom image
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Status

```bash
curl http://localhost:8000/models
```

### Inference

```bash
curl -X POST http://localhost:8000/inference \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@input.wav" \
  --output response.wav
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | Hugging Face token for gated model access |
| `API_KEY` | No | `default-api-key-change-me` | API key for authentication |
| `MODEL_REPO` | No | `nvidia/personaplex-7b-v1` | Hugging Face model repository |
| `DEVICE` | No | `cuda` | Device to run inference on |
| `MAX_AUDIO_SIZE` | No | `10485760` | Max audio file size in bytes |

## Performance Notes

- Model size: ~17GB
- Recommended GPU: RTX 4090 (24GB) or A100 (40GB/80GB)
- Cold start: Model loading takes ~2-5 minutes
- Expected latency: 2-5 seconds for short audio clips

## Troubleshooting

### Model fails to load
- Verify HF_TOKEN is set and valid
- Ensure you accepted the model license on HF
- Check GPU memory requirements

### Inference errors
- Ensure model is loaded (check /models endpoint)
- Verify audio file is WAV format
- Check audio file size is under MAX_AUDIO_SIZE
# personaplex-api
