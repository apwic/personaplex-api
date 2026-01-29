"""Modal deployment for PersonaPlex API."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends
from pydantic import BaseModel
from modal import App, Image, Secret, enter, exit, web_server

MODEL_REPO = os.getenv("MODEL_REPO", "nvidia/personaplex-7b-v1")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")

app = App("personaplex-api")

image = Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install(
    "moshi>=0.2.12",
    "torch>=2.2.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "python-multipart>=0.0.6",
    "huggingface-hub>=0.24.0",
    "pydantic>=2.0.0",
    "aiofiles>=23.0.0",
    "torchaudio>=2.2.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "loguru>=0.7.0",
)


def load_model():
    """Load Moshi model from Hugging Face."""
    import torch
    from moshi.models import loaders

    print(f"Loading model from {MODEL_REPO}...")

    mimi_model = loaders.get_mimi(device="cuda", dtype=torch.bfloat16)
    mimi_model.set_num_codebooks(8)

    moshi_model = loaders.get_moshi_lm(
        None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16
    )

    print("Model loaded successfully!")
    return moshi_model, mimi_model


moshi_model = None
mimi_model = None


@app.cls(
    image=image,
    gpu="A100",
    scaledown_window=300,
    secrets=[
        Secret.from_name("huggingface-secret"),
        Secret.from_name("api-key"),
    ],
    timeout=3600,
)
class ModelAPI:
    @enter()
    def load_models(self):
        global moshi_model, mimi_model
        moshi_model, mimi_model = load_model()

    @exit()
    def cleanup(self):
        global moshi_model, mimi_model
        moshi_model = None
        mimi_model = None

    @web_server(port=8000)
    def serve(self):
        import uvicorn
        uvicorn.run(self._fastapi_app, host="0.0.0.0", port=8000)

    @property
    def _fastapi_app(self):
        fastapi_app = FastAPI(
            title="PersonaPlex Speech-to-Speech API",
            description="Real-time speech-to-speech inference using nvidia/personaplex-7b-v1",
            version="1.0.0"
        )

        MAX_AUDIO_SIZE = 10485760

        class ModelStatus(BaseModel):
            model_id: str
            status: str
            device: str
            error: str | None = None

        class HealthResponse(BaseModel):
            status: str
            model_loaded: bool

        async def verify_api_key(authorization: str = Header(None)):
            if not authorization:
                raise HTTPException(status_code=401, detail="Missing Authorization header")
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid Authorization format")
            token = authorization[7:]
            if token != API_KEY:
                raise HTTPException(status_code=403, detail="Invalid API key")
            return token

        @fastapi_app.get("/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="healthy" if moshi_model is not None else "degraded",
                model_loaded=moshi_model is not None
            )

        @fastapi_app.get("/models", response_model=ModelStatus)
        async def get_model_status():
            return ModelStatus(
                model_id=MODEL_REPO,
                status="loaded" if moshi_model is not None else "error",
                device="cuda",
                error=None if moshi_model is not None else "Model not loaded"
            )

        @fastapi_app.post("/inference")
        async def inference(
            audio: UploadFile = File(...),
            _auth: str = Depends(verify_api_key)
        ):
            if not audio.filename or not audio.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="Only WAV files are supported")

            content = await audio.read()
            if len(content) > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large. Max size: {MAX_AUDIO_SIZE} bytes"
                )

            if moshi_model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                import io
                import tempfile
                import torch
                import torchaudio
                from fastapi.responses import Response

                waveform, orig_sr = torchaudio.load(io.BytesIO(content))
                if orig_sr != 24000:
                    resampler = torchaudio.transforms.Resample(orig_sr, 24000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                with torch.no_grad():
                    frames = mimi_model.encode(waveform.to("cuda"))
                    audio_out = moshi_model.generate(frames)
                    output = mimi_model.decode(audio_out)

                buffer = io.BytesIO()
                torchaudio.save(buffer, output.cpu(), 24000, format="wav")

                return Response(
                    content=buffer.getvalue(),
                    media_type="audio/wav",
                    headers={"Content-Disposition": 'attachment; filename="response.wav"'}
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        return fastapi_app


if __name__ == "__main__":
    app.deploy()
