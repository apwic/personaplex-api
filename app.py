import os
import tempfile
import torch
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from moshi.models import loaders

app = FastAPI(
    title="PersonaPlex Speech-to-Speech API",
    description="Real-time speech-to-speech inference using nvidia/personaplex-7b-v1",
    version="1.0.0"
)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
MODEL_REPO = os.getenv("MODEL_REPO", "nvidia/personaplex-7b-v1")
DEVICE = os.getenv("DEVICE", "cuda")
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 10485760))  # 10MB default

# Model storage
MODEL_CACHE_DIR = Path("/tmp/moshi-models")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global model references
moshi_model = None
mimi_model = None


class ModelStatus(BaseModel):
    model_id: str
    status: str
    device: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


async def verify_api_key(authorization: str = Header(None)):
    """Verify API key from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = authorization[7:]  # Remove "Bearer " prefix
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return token


def load_model():
    """Load Moshi model from Hugging Face."""
    global moshi_model, mimi_model

    try:
        print(f"Loading model from {MODEL_REPO}...")

        # Set Hugging Face token for gated model access
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN

        # Download and load the language model
        moshi_model = loaders.get_moshi_lm(
            None,  # Will download from HF
            device=torch.device(DEVICE),
            dtype=torch.bfloat16
        )

        # Download and load the Mimi codec model
        mimi_model = loaders.get_mimi(
            device=torch.device(DEVICE),
            dtype=torch.bfloat16
        )

        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_model()
    if not success:
        print("Warning: Model failed to load. Inference will not be available.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if moshi_model is not None else "degraded",
        model_loaded=moshi_model is not None
    )


@app.get("/models", response_model=ModelStatus)
async def get_model_status():
    """Get current model status."""
    return ModelStatus(
        model_id=MODEL_REPO,
        status="loaded" if moshi_model is not None else "error",
        device=DEVICE,
        error=None if moshi_model is not None else "Model not loaded"
    )


@app.post("/inference")
async def inference(
    audio: UploadFile = File(...),
    _auth: str = Depends(verify_api_key)
):
    """
    Process audio through PersonaPlex model.

    Accepts WAV audio, returns AI-generated speech audio.
    """
    # Validate file type
    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Validate file size
    content = await audio.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Max size: {MAX_AUDIO_SIZE} bytes"
    )

    if moshi_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # TODO: Implement actual Moshi inference
            # This requires understanding the exact Moshi pipeline API
            # For now, return a placeholder response

            # The actual implementation would:
            # 1. Load audio using mimi_model for encoding
            # 2. Process through moshi_model
            # 3. Decode output with mimi_model
            # 4. Return audio

            raise HTTPException(
                status_code=501,
                detail="Inference not yet implemented. Moshi pipeline integration needed."
            )

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
