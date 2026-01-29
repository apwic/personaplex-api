"""Deploy PersonaPlex to Replicate using Python SDK."""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

import replicate

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_TOKEN:
    raise ValueError("Set REPLICATE_API_TOKEN environment variable")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN

print("Deploying PersonaPlex to Replicate...")

try:
    model = replicate.models.create(
        owner="anca",
        name="personaplex",
        visibility="public",
        hardware="A100",
        description="Real-time speech-to-speech conversational AI using nvidia/personaplex-7b-v1",
    )
    print(f"Model created: {model.url}")

    # Get the version string from predict.py
    print("\nTo deploy a version, push your code to GitHub and:")
    print("1. Go to https://replicate.com/create")
    print("2. Select your GitHub repository")
    print("3. Set entry point to: predict.py:predict")
    print("4. Configure GPU: A100, Memory: 16GB")

except replicate.error.ReplicateError as e:
    print(f"Error: {e}")
    print("\nAlternative: Push to GitHub and create via web UI at https://replicate.com/create")
