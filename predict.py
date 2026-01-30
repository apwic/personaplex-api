"""Replicate prediction module for PersonaPlex speech-to-speech API."""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torchaudio
from cog import BasePredictor, Input, Path as CogPath, Secret


class Predictor(BasePredictor):
    """PersonaPlex speech-to-speech predictor using Moshi and Mimi."""

    def setup(self):
        """Load models during setup."""
        print("[SETUP] Starting model loading...", flush=True)
        
        try:
            from moshi.models import loaders, LMGen
            import huggingface_hub
            
            token = os.getenv("HF_TOKEN")
            if token:
                print("[SETUP] Logging in to HuggingFace...", flush=True)
                huggingface_hub.login(token=token)
            
            print("[SETUP] Loading checkpoint info from kyutai/moshiko-pytorch-bf16...", flush=True)
            self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
            
            print("[SETUP] Loading Mimi model...", flush=True)
            self.mimi = self.checkpoint_info.get_mimi(device="cuda")
            self.mimi.eval()
            
            print("[SETUP] Loading text tokenizer...", flush=True)
            self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()
            
            print("[SETUP] Loading Moshi LM...", flush=True)
            self.lm = self.checkpoint_info.get_moshi(device="cuda", dtype=torch.bfloat16)
            self.lm.eval()
            
            print("[SETUP] Creating LMGen...", flush=True)
            self.lm_gen = LMGen(
                self.lm,
                use_sampling=True,
                temp=0.8,
                temp_text=0.7,
            )
            
            self.sample_rate = 24000
            self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            print(f"[SETUP] Complete. Frame size: {self.frame_size}", flush=True)
            
        except Exception as e:
            print(f"[SETUP ERROR] {e}", flush=True)
            traceback.print_exc()
            raise

    def predict(
        self,
        audio: CogPath = Input(description="Input audio file (WAV format)"),
        hf_token: Secret = Input(description="HF token (optional)")
    ) -> CogPath:
        print(f"[PREDICT] Starting prediction with audio: {audio}", flush=True)
        
        try:
            # Load audio
            print("[PREDICT] Loading audio file...", flush=True)
            waveform, orig_sr = torchaudio.load(str(audio))
            print(f"[PREDICT] Loaded: shape={waveform.shape}, sr={orig_sr}", flush=True)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if orig_sr != self.sample_rate:
                print(f"[PREDICT] Resampling {orig_sr} -> {self.sample_rate}", flush=True)
                resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Prepare model state
            print("[PREDICT] Setting up streaming...", flush=True)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)
            
            # Move to device and add batch dimension: (1, channels, samples)
            waveform = waveform.unsqueeze(0).to("cuda")
            print(f"[PREDICT] Waveform shape: {waveform.shape}, dtype: {waveform.dtype}", flush=True)
            
            # Ensure float32
            if waveform.dtype != torch.float32:
                waveform = waveform.float()
                print(f"[PREDICT] Converted to float32", flush=True)
            
            # Process audio - match official run_inference.py pattern
            print("[PREDICT] Processing audio...", flush=True)
            output_chunks = []
            first_frame = True
            
            # Split into frame-sized chunks (matching run_inference.py)
            chunks = [
                chunk for chunk in waveform.split(self.frame_size, dim=2)
                if chunk.shape[-1] == self.frame_size
            ]
            print(f"[PREDICT] {len(chunks)} chunks to process", flush=True)
            
            with torch.no_grad():
                for i, chunk in enumerate(chunks):
                    print(f"[PREDICT] Processing chunk {i+1}/{len(chunks)}", flush=True)
                    
                    # Encode
                    codes = self.mimi.encode(chunk)
                    print(f"[PREDICT] Encoded codes shape: {codes.shape}", flush=True)
                    
                    # First frame handling (from run_inference.py)
                    if first_frame:
                        tokens = self.lm_gen.step(codes)
                        first_frame = False
                        if tokens is None:
                            print("[PREDICT] First frame returned None (expected)", flush=True)
                            continue
                    else:
                        tokens = self.lm_gen.step(codes)
                        if tokens is None:
                            print("[PREDICT] tokens is None, continuing", flush=True)
                            continue
                    
                    print(f"[PREDICT] tokens shape: {tokens.shape}", flush=True)
                    
                    # Decode audio (skip text token at index 0)
                    if tokens.shape[1] > 1:
                        audio_tokens = tokens[:, 1:]  # (batch, dep_q, frames)
                        print(f"[PREDICT] Decoding audio tokens: {audio_tokens.shape}", flush=True)
                        output_chunk = self.mimi.decode(audio_tokens)
                        print(f"[PREDICT] Decoded chunk: {output_chunk.shape}", flush=True)
                        output_chunks.append(output_chunk.cpu())
            
            print(f"[PREDICT] Generated {len(output_chunks)} chunks", flush=True)
            
            # Concatenate output
            if output_chunks:
                output = torch.cat(output_chunks, dim=-1)
            else:
                print("[PREDICT] Warning: no output, using silence", flush=True)
                output = torch.zeros(1, self.sample_rate)
            
            print(f"[PREDICT] Final output: {output.shape}", flush=True)
            
            # Save
            output_path = Path(tempfile.mktemp(suffix=".wav"))
            torchaudio.save(str(output_path), output, self.sample_rate)
            print(f"[PREDICT] Saved to {output_path}", flush=True)
            
            return CogPath(output_path)
            
        except Exception as e:
            error_msg = f"[PREDICT ERROR] {type(e).__name__}: {e}"
            print(error_msg, flush=True)
            print(traceback.format_exc(), flush=True)
            # Re-raise with more context
            raise RuntimeError(error_msg) from e
