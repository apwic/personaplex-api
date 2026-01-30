"""Audio processing utilities for PersonaPlex API."""

import io
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple

SAMPLE_RATE = 24000


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample to Moshi native sample rate.

    Args:
        file_path: Path to audio file (WAV format)

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    waveform, orig_sr = torchaudio.load(file_path)

    if orig_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, SAMPLE_RATE


def encode_audio(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Encode audio: int = tensor to WAV bytes.

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate

    Returns:
        WAV bytes
    """
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    return buffer.getvalue()


def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Load audio from bytes and resample to native rate.

    Args:
        audio_bytes: WAV audio data

    Returns:
        Audio tensor
    """
    waveform, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))

    if orig_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform
