"""Replicate prediction module for PersonaPlex speech-to-speech API.

This module is loaded by Replicate to serve predictions.
"""

import os
import io
import tempfile
from pathlib import Path

import torch
import torchaudio


def load_model():
    """Load Moshi LM and Mimi codec models.

    Returns:
        Tuple of (moshi_model, mimi_model)
    """
    from moshi.models import loaders

    mimi_model = loaders.get_mimi(
        device="cuda",
        dtype=torch.bfloat16
    )
    mimi_model.set_num_codebooks(8)

    moshi_model = loaders.get_moshi_lm(
        None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16
    )

    return moshi_model, mimi_model


def predict(
    audio: bytes = None,
) -> bytes:
    """Run speech-to-speech inference.

    Args:
        audio: WAV audio bytes input

    Returns:
        WAV audio bytes output

    Raises:
        ValueError: If no audio provided
    """
    if audio is None:
        raise ValueError("audio input is required")

    # Load model (cached across calls by Replicate)
    if not hasattr(predict, "_models_loaded"):
        predict.moshi_model, predict.mimi_model = load_model()
        predict._models_loaded = True

    moshi_model = predict.moshi_model
    mimi_model = predict.mimi_model

    # Load and resample audio
    waveform, orig_sr = torchaudio.load(io.BytesIO(audio))
    if orig_sr != 24000:
        resampler = torchaudio.transforms.Resample(orig_sr, 24000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Run inference
    with torch.no_grad():
        frames = mimi_model.encode(waveform.to("cuda"))
        audio_out = moshi_model.generate(frames)
        output = mimi_model.decode(audio_out)

    # Save to bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, output.cpu(), 24000, format="wav")
    return buffer.getvalue()
