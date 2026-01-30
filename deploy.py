"""Modal deployment for PersonaPlex API with WebSocket streaming."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from modal import App, Image, Secret, enter, exit, asgi_app

MODEL_REPO = os.getenv("MODEL_REPO", "nvidia/personaplex-7b-v1")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "4"))

app = App("personaplex-api")

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libopus-dev", "git")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-multipart>=0.0.6",
        "huggingface-hub>=0.24.0",
        "pydantic>=2.0.0",
        "aiofiles>=23.0.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "loguru>=0.7.0",
        "websockets>=12.0",
    )
    # Install moshi from NVIDIA's PersonaPlex repo (has modified loaders)
    .run_commands("pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi")
)


def load_model():
    """Load PersonaPlex model from NVIDIA HuggingFace repo."""
    import torch
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download

    print("Loading PersonaPlex from nvidia/personaplex-7b-v1...")

    # Download model files from HuggingFace
    mimi_path = hf_hub_download(
        repo_id="nvidia/personaplex-7b-v1",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    lm_path = hf_hub_download(
        repo_id="nvidia/personaplex-7b-v1",
        filename="model.safetensors"
    )

    print(f"Downloaded mimi: {mimi_path}")
    print(f"Downloaded LM: {lm_path}")

    # Load models using NVIDIA's loaders
    mimi_model = loaders.get_mimi(mimi_path, device="cuda")
    moshi_model = loaders.get_moshi_lm(lm_path, device="cuda", dtype=torch.bfloat16)

    mimi_model.eval()
    moshi_model.eval()

    print("PersonaPlex model loaded successfully!")
    return moshi_model, mimi_model


# Default persona configuration
DEFAULT_TEXT_PROMPT = """You are a helpful, friendly AI assistant. You engage in natural conversation,
answer questions clearly, and help users with their requests. You speak in a warm and professional tone."""

DEFAULT_VOICE = "NATF2"  # Natural female voice 2


moshi_model = None
mimi_model = None
lm_gen = None


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
        global moshi_model, mimi_model, lm_gen
        from moshi.models import LMGen

        moshi_model, mimi_model = load_model()
        # Enter streaming mode once - stays active for container lifetime
        mimi_model.streaming_forever(1)
        # Create LMGen and enter its streaming mode (handles moshi_model streaming)
        lm_gen = LMGen(moshi_model, device="cuda", use_sampling=True, temp=0.8, temp_text=0.7)
        lm_gen.streaming_forever(1)

    @exit()
    def cleanup(self):
        global moshi_model, mimi_model, lm_gen
        moshi_model = None
        mimi_model = None
        lm_gen = None

    @asgi_app()
    def serve(self):
        return self._fastapi_app

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

        # WebSocket streaming state
        streaming_sessions = {}
        gpu_lock = asyncio.Lock()

        async def verify_ws_auth(websocket: WebSocket) -> str | None:
            """Verify WebSocket auth via query param or first message."""
            token = websocket.query_params.get("token")
            if not token:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                    msg = json.loads(data)
                    if msg.get("type") != "auth":
                        await websocket.close(code=4001, reason="Authentication required")
                        return None
                    token = msg.get("token")
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    await websocket.close(code=4001, reason="Authentication failed")
                    return None

            if token != API_KEY:
                await websocket.close(code=4003, reason="Invalid token")
                return None

            return websocket.query_params.get("session_id") or str(uuid.uuid4())

        def create_streaming_session(session_id: str):
            """Create a new streaming session using global LMGen."""
            import torch
            import struct

            class StreamingSession:
                def __init__(self):
                    self.session_id = session_id
                    self.frame_size = 1920

                def start(self):
                    pass

                def cleanup(self):
                    # Reset streaming state for next session
                    try:
                        mimi_model.reset_streaming()
                        lm_gen.reset_streaming()
                    except (AssertionError, AttributeError):
                        pass

                def process_frame(self, audio_bytes: bytes) -> list[bytes]:
                    samples = struct.unpack(f"<{self.frame_size}h", audio_bytes)
                    waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
                    waveform = waveform.unsqueeze(0).unsqueeze(0).to("cuda")

                    output_chunks = []
                    with torch.no_grad():
                        codes = mimi_model.encode(waveform)
                        print(f"Encoded codes shape: {codes.shape}")

                        for c in range(codes.shape[-1]):
                            tokens = lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                print(f"  Step {c}: tokens is None")
                                continue
                            print(f"  Step {c}: tokens shape {tokens.shape}")
                            if tokens.shape[1] > 1:
                                audio_tokens = tokens[:, 1:]
                                output_chunk = mimi_model.decode(audio_tokens)
                                pcm = (output_chunk.squeeze().cpu() * 32768.0).clamp(-32768, 32767).to(torch.int16)
                                output_chunks.append(pcm.numpy().tobytes())

                    print(f"Output chunks: {len(output_chunks)}, sizes: {[len(c) for c in output_chunks]}")
                    return output_chunks

            return StreamingSession()

        async def send_event(websocket: WebSocket, event: str, payload: dict = None):
            msg = {"type": "event", "event": event, "payload": payload or {}, "timestamp": time.time()}
            await websocket.send_text(json.dumps(msg))

        @fastapi_app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            """Bidirectional audio streaming endpoint."""
            await websocket.accept()

            session_id = await verify_ws_auth(websocket)
            if session_id is None:
                return

            if len(streaming_sessions) >= MAX_CONCURRENT_SESSIONS:
                await websocket.close(code=4008, reason="Session limit reached")
                return

            if moshi_model is None:
                await websocket.close(code=4001, reason="Model not loaded")
                return

            async with gpu_lock:
                session = create_streaming_session(session_id)
                session.start()
                streaming_sessions[session_id] = session

            await send_event(websocket, "session_started", {"session_id": session_id, "frame_size": 1920})

            try:
                while True:
                    message = await websocket.receive()

                    if message["type"] == "websocket.disconnect":
                        break

                    if "bytes" in message and message["bytes"]:
                        try:
                            t0 = time.time()
                            async with gpu_lock:
                                output_chunks = session.process_frame(message["bytes"])
                            t1 = time.time()
                            for chunk in output_chunks:
                                await websocket.send_bytes(chunk)
                            if output_chunks and (t1 - t0) > 0.1:
                                print(f"Frame processing took {(t1-t0)*1000:.0f}ms, {len(output_chunks)} chunks")
                        except Exception as e:
                            await send_event(websocket, "error", {"code": "E003", "message": str(e)})

                    elif "text" in message and message["text"]:
                        try:
                            msg = json.loads(message["text"])
                            action = msg.get("action")

                            if action == "reset":
                                async with gpu_lock:
                                    session.start()
                                await send_event(websocket, "reset_ack")
                            elif action == "stop":
                                await send_event(websocket, "session_ended")
                                break
                            elif action == "ping":
                                await send_event(websocket, "pong", {"server_time": time.time()})
                        except json.JSONDecodeError:
                            pass

            except WebSocketDisconnect:
                pass
            finally:
                session = streaming_sessions.get(session_id)
                if session:
                    async with gpu_lock:
                        session.cleanup()
                    del streaming_sessions[session_id]

        return fastapi_app


if __name__ == "__main__":
    app.deploy()
