"""Streaming session management for real-time audio processing."""

import asyncio
import struct
import uuid
from typing import Optional

import torch
from loguru import logger


class StreamingSession:
    """Per-connection streaming state for Moshi inference."""

    def __init__(
        self,
        session_id: str,
        mimi_model,
        moshi_model,
        sample_rate: int = 24000,
        frame_rate: int = 50,
    ):
        from moshi.models import LMGen

        self.session_id = session_id
        self.mimi_model = mimi_model
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frame_size = sample_rate // frame_rate

        self.lm_gen = LMGen(
            moshi_model,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
        )

        self.is_streaming = False
        self.frames_processed = 0
        self._skip_first_frame = True

    def start(self):
        """Initialize streaming state."""
        self.mimi_model.reset_streaming()
        self.lm_gen.reset_streaming()
        self.mimi_model.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.is_streaming = True
        self._skip_first_frame = True
        self.frames_processed = 0
        logger.info(f"Session {self.session_id}: streaming started")

    def reset(self):
        """Reset streaming state, clearing context."""
        self.start()
        logger.info(f"Session {self.session_id}: streaming reset")

    def stop(self):
        """End streaming session."""
        self.is_streaming = False
        logger.info(f"Session {self.session_id}: streaming stopped, processed {self.frames_processed} frames")

    def process_frame(self, audio_bytes: bytes) -> list[bytes]:
        """Process single audio frame, return generated audio chunks.

        Args:
            audio_bytes: Raw PCM audio (16-bit signed mono, 24kHz)
                        Expected size: frame_size * 2 bytes (960 bytes for 480 samples)

        Returns:
            List of output audio byte chunks
        """
        if not self.is_streaming:
            raise RuntimeError("Session not started")

        expected_size = self.frame_size * 2
        if len(audio_bytes) != expected_size:
            logger.warning(
                f"Session {self.session_id}: expected {expected_size} bytes, got {len(audio_bytes)}"
            )

        samples = struct.unpack(f"<{self.frame_size}h", audio_bytes)
        waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0).to("cuda")

        output_chunks = []

        with torch.no_grad():
            codes = self.mimi_model.encode(waveform)

            if self._skip_first_frame:
                self.mimi_model.reset_streaming()
                self._skip_first_frame = False
                self.frames_processed += 1
                return output_chunks

            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue

                if tokens.shape[1] > 1:
                    audio_tokens = tokens[:, 1:]
                    output_chunk = self.mimi_model.decode(audio_tokens)
                    pcm = (output_chunk.squeeze().cpu() * 32768.0).clamp(-32768, 32767).to(torch.int16)
                    output_chunks.append(pcm.numpy().tobytes())

        self.frames_processed += 1
        return output_chunks


def create_session(
    mimi_model,
    moshi_model,
    session_id: Optional[str] = None,
) -> StreamingSession:
    """Create a new streaming session."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    return StreamingSession(session_id, mimi_model, moshi_model)
