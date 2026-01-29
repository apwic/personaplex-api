"""Replicate prediction module for PersonaPlex speech-to-speech API.

This module is loaded by Replicate to serve predictions using the Cog framework.
"""

import os
import tempfile
from pathlib import Path

import torch
import torchaudio
import cog


class Predictor(cog.Predictor):
    """PersonaPlex speech-to-speech predictor using Moshi and Mimi."""

    def setup(self):
        """Load Moshi LM and Mimi codec models.

        This runs once when the container starts. Models are cached in memory
        for subsequent predictions.
        """
        from moshi.models import loaders

        # Set Hugging Face token for gated model access
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)

        # Load Mimi codec for audio encoding/decoding
        self.mimi_model = loaders.get_mimi(
            device="cuda",
            dtype=torch.bfloat16
        )
        self.mimi_model.set_num_codebooks(8)

        # Load Moshi language model for speech generation
        self.moshi_model = loaders.get_moshi_lm(
            None,
            device=torch.device("cuda"),
            dtype=torch.bfloat16
        )

        # Set models to evaluation mode
        self.mimi_model.eval()
        self.moshi_model.eval()

        self.sample_rate = 24000

    @cog.input(
        "audio",
        type=cog.File,
        description="Input audio file (WAV format, will be resampled to 24kHz if needed)",
        help="Upload a WAV audio file to generate a speech response."
    )
    def predict(self, audio):
        """Run speech-to-speech inference.

        Args:
            audio: Input audio file (WAV format)

        Returns:
            Audio file (WAV format) - generated speech response
        """
        # Load audio file
        waveform, orig_sr = torchaudio.load(audio.path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Move to GPU
        waveform = waveform.to("cuda")

        # Encode audio to frames using Mimi
        with torch.no_grad():
            frames = self.mimi_model.encode(waveform)

            # Generate speech frames using Moshi LM
            audio_out = self.moshi_model.generate(frames)

            # Decode frames back to audio using Mimi
            output = self.mimi_model.decode(audio_out)

        # Save to temporary file and return
        output_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(output_path, output.cpu(), self.sample_rate)

        return cog.File(output_path)
