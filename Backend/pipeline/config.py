"""Configuration module for the AI Video Inpainting Pipeline.

This module centralizes all file paths, model identifiers, hyperparameters,
and hardware optimization settings (such as VRAM management for Stable Video Diffusion)
used across the AI video inpainting pipeline.
"""

import os
from pathlib import Path
from typing import Tuple

# Attempt to load environment variables from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    # python-dotenv is optional; continue silently if not installed
    pass

# ==============================================================================
# Directory Paths
# ==============================================================================

# Base directory for the project (directory containing this config file)
BASE_DIR: Path = Path(__file__).resolve().parent

# Directory for temporary files (frames, intermediate outputs, cache)
TEMP_DIR: Path = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==============================================================================
# Model Configurations & Hyperparameters
# ==============================================================================

# Vision-Language Model identifier for Ollama (used for scene understanding / prompting)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava")

# HuggingFace model repository ID for Stable Video Diffusion (SVD)
SVD_MODEL_ID: str = os.getenv(
    "SVD_MODEL", "stabilityai/stable-video-diffusion-img2vid-xt"
)

# Number of video frames to generate per chunk/inference
# SVD-XT is natively fine-tuned on 25 frames
SVD_NUM_FRAMES: int = int(os.getenv("SVD_NUM_FRAMES", "25"))

# Number of frames to decode at a time through the VAE
# Setting decode_chunk_size=1 decodes one frame at a time to drastically reduce
# peak VRAM consumption, minimizing memory usage for consumer GPUs (e.g. 6GB VRAM)
SVD_DECODE_CHUNK_SIZE: int = int(os.getenv("SVD_DECODE_CHUNK_SIZE", "1"))

# Target frames per second (FPS) for the generated video output
SVD_FPS: int = int(os.getenv("SVD_FPS", "7"))

# Native training resolution for Stable Video Diffusion (width, height)
# (1024, 576) matches the standard 16:9 native aspect ratio and pretraining resolution
SVD_RESOLUTION: Tuple[int, int] = (1024, 576)

# Random seed for reproducible video generation across pipeline runs
SVD_SEED: int = int(os.getenv("SVD_SEED", "42"))
