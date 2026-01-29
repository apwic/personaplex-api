"""PersonaPlex Speech-to-Speech API.

Real-time speech-to-speech inference using nvidia/personaplex-7b-v1.
"""

import os
import tempfile
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import Response
from pydantic import BaseModel

from services import (
    verify_api_key,
    load_model,
    is_model_loaded,
    run_model_inference,
    load_audio_from_bytes,
    encode_audio,
    SAMPLE_RATE
)

app = FastAPI(
    title="PersonaPlex Speech-to-Speech API",
    description="Real-time speech-to-speech inference using nvidia/personaplex-7b-v1",
    version="1.0.0"
)

API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
MODEL_REPO = os.getenv("MODEL_REPO", "nvidia/personaplex-7b-v1")
DEVICE = os.getenv("DEVICE", "cuda")
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 10485760))


class ModelStatus(BaseModel):
    model_id: str
    status: str
    device: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    if not load_model():
        print("Warning: Model failed to load. Inference will not be available.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded()
    )


@app.get("/models", response_model=ModelStatus)
async def get_model_status():
    """Get current model status."""
    return ModelStatus(
        model_id=MODEL_REPO,
        status="loaded" if is_model_loaded() else "error",
        device=DEVICE,
        error=None if is_model_loaded() else "Model not loaded"
    )


@app.post("/inference")
async def inference(
    audio: UploadFile = File(...),
    _auth: str = Depends(verify_api_key)
):
    """Process audio through PersonaPlex model.

    Accepts WAV audio, returns AI-generated speech audio.
    """
    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    content = await audio.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Max size: {MAX_AUDIO_SIZE} bytes"
        )

    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_audio = load_audio_from_bytes(content)
        output_audio = run_model_inference(input_audio)
        output_bytes = encode_audio(output_audio, SAMPLE_RATE)

        return Response(
            content=output_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="response.wav"'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
