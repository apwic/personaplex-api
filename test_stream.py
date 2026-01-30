"""Test client for PersonaPlex WebSocket streaming."""

import asyncio
import json
import struct
import wave
import sys
from pathlib import Path

import websockets

API_URL = "wss://apwic--personaplex-api-modelapi-serve.modal.run/ws/stream"
API_TOKEN = "pa_dasklhuozchxvwqjeljAfdja"
FRAME_SIZE = 1920  # 80ms at 24kHz (Mimi requirement)
SAMPLE_RATE = 24000


async def stream_audio(input_wav: str, output_wav: str = "output.wav"):
    """Stream audio file through the API and save response."""

    # Read input audio
    print(f"Reading {input_wav}...")
    with wave.open(input_wav, 'rb') as wf:
        assert wf.getnchannels() == 1, "Input must be mono"
        assert wf.getsampwidth() == 2, "Input must be 16-bit"
        orig_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    # Resample if needed (simple approach - for production use proper resampling)
    if orig_rate != SAMPLE_RATE:
        print(f"Warning: Input is {orig_rate}Hz, expected {SAMPLE_RATE}Hz. Results may vary.")

    # Convert to samples
    samples = struct.unpack(f"<{len(audio_data)//2}h", audio_data)

    # Split into frames
    frames = []
    for i in range(0, len(samples) - FRAME_SIZE + 1, FRAME_SIZE):
        frame = samples[i:i + FRAME_SIZE]
        frames.append(struct.pack(f"<{FRAME_SIZE}h", *frame))

    print(f"Split into {len(frames)} frames")

    # Connect and stream
    uri = f"{API_URL}?token={API_TOKEN}"
    output_chunks = []

    async with websockets.connect(uri) as ws:
        # Wait for session start
        msg = json.loads(await ws.recv())
        print(f"Session started: {msg['payload']['session_id']}")

        # Send frames and collect responses
        for i, frame in enumerate(frames):
            await ws.send(frame)

            # Check for response (non-blocking)
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    if isinstance(response, bytes):
                        output_chunks.append(response)
                        print(f"Received {len(response)} bytes of audio")
                    else:
                        msg = json.loads(response)
                        if msg.get("event") == "error":
                            print(f"Error: {msg['payload']}")
            except asyncio.TimeoutError:
                pass

            if (i + 1) % 50 == 0:
                print(f"Sent {i + 1}/{len(frames)} frames...")

        # Wait for remaining responses
        print("Waiting for remaining responses...")
        await asyncio.sleep(1)

        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                if isinstance(response, bytes):
                    output_chunks.append(response)
                    print(f"Received {len(response)} bytes of audio")
        except asyncio.TimeoutError:
            pass

        # Stop session
        await ws.send(json.dumps({"action": "stop"}))

    # Save output
    if output_chunks:
        output_data = b"".join(output_chunks)
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(output_data)
        print(f"Saved {len(output_data)} bytes to {output_wav}")
    else:
        print("No audio output received")


def create_test_audio(filename: str = "test_input.wav", duration: float = 2.0):
    """Create a simple test tone."""
    import math

    num_samples = int(SAMPLE_RATE * duration)
    samples = []

    # Generate 440Hz sine wave
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        sample = int(16000 * math.sin(2 * math.pi * 440 * t))
        samples.append(sample)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    print(f"Created test audio: {filename}")
    return filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_stream.py <input.wav> [output.wav]")
        print("       python test_stream.py --create-test")
        sys.exit(1)

    if sys.argv[1] == "--create-test":
        input_file = create_test_audio()
        output_file = "test_output.wav"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

    asyncio.run(stream_audio(input_file, output_file))
