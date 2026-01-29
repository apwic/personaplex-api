"""Moshi model service for PersonaPlex API."""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from moshi.models import loaders


class MoshiModelService:
    """Service for loading and running Moshi model inference."""

    def __init__(
        self,
        model_repo: str = "nvidia/personaplex-7b-v1",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None
    ):
        self.model_repo = model_repo
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if dtype == "bfloat16" else torch.float16
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/moshi-models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.moshi_model: Optional[loaders.LMModel] = None
        self.mimi_model: Optional[loaders.Mimi] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded and self.moshi_model is not None

    def load(self) -> bool:
        """Load Moshi LM and Mimi codec models from Hugging Face.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.model_repo}...")

            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            self.mimi_model = loaders.get_mimi(
                device=self.device,
                dtype=self.dtype
            )
            self.mimi_model.set_num_codebooks(8)

            self.moshi_model = loaders.get_moshi_lm(
                None,
                device=self.device,
                dtype=self.dtype
            )

            self._loaded = True
            logger.info("Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def run_inference(
        self,
        input_audio: torch.Tensor
    ) -> torch.Tensor:
        """Run speech-to-speech inference.

        Args:
            input_audio: Input audio tensor (1, samples)

        Returns:
            Output audio tensor (1, samples)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            if self.device.type == "cuda":
                input_audio = input_audio.to(self.device)

            frames = self.mimi_model.encode(input_audio)
            audio = self.moshi_model.generate(frames)
            output = self.mimi_model.decode(audio)

        return output.cpu()

    def unload(self):
        """Unload models and free memory."""
        del self.moshi_model
        del self.mimi_model
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        self._loaded = False
        logger.info("Models unloaded")


# Global service instance
_model_service: Optional[MoshiModelService] = None


def get_model_service() -> MoshiModelService:
    """Get or create the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = MoshiModelService(
            model_repo=os.getenv("MODEL_REPO", "nvidia/personaplex-7b-v1"),
            device=os.getenv("DEVICE", "cuda"),
            dtype=os.getenv("DTYPE", "bfloat16"),
            cache_dir=os.getenv("MODEL_CACHE_DIR")
        )
    return _model_service


def load_model() -> bool:
    """Load the global model service."""
    service = get_model_service()
    return service.load()


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    service = get_model_service()
    return service.is_loaded


def run_model_inference(input_audio: torch.Tensor) -> torch.Tensor:
    """Run inference through the global model service."""
    service = get_model_service()
    return service.run_inference(input_audio)
