# Deploying to Replicate

This guide covers how to deploy PersonaPlex to Replicate for pay-per-use inference.

## Prerequisites

1. Replicate account: https://replicate.com
2. Hugging Face account with access to `nvidia/personaplex-7b-v1`
3. Replicate CLI installed and authenticated

## Setup

```bash
# Install Replicate CLI
pip install replicate

# Login (opens browser)
replicate login
```

## Deploy

### Option 1: Push from this repository

```bash
replicate create --yaml replicate.yaml
```

This will:
1. Build a container with your requirements
2. Upload to Replicate's infrastructure
3. Return a model URL like `replicate.com/your-username/personaplex`

### Option 2: Using the web interface

1. Go to https://replicate.com/create
2. Select "Start from a GitHub repository"
3. Enter your fork URL
4. Configure GPU (A100) and memory (16GB)
5. Click "Create"

## Usage

Once deployed, call the model:

```python
import replicate

output = replicate.run(
    "your-username/personaplex",
    input={"audio": open("input.wav", "rb").read()}
)

with open("response.wav", "wb") as f:
    f.write(output)
```

## Environment Variables

Set your Hugging Face token in Replicate:

```bash
replicate env set HF_TOKEN=hf_xxxxx
```

## Troubleshooting

### Model fails to load
- Ensure `HF_TOKEN` is set and has access to the gated model
- Check the model fits in 16GB GPU memory

### Audio processing errors
- Input must be WAV format
- Maximum file size depends on your Replicate tier

### Cold starts
- Replicate keeps models warm for a few minutes after use
- First call may take 1-2 minutes to load the model
