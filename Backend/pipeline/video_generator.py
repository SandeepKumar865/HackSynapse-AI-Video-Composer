"""Video Generator Module.

This module generates a bridge video from a start frame using Stable Video Diffusion
(SVD-XT) via HuggingFace diffusers, specifically optimized for 6GB VRAM GPUs (such as
NVIDIA GeForce RTX 4050 Mobile / RTX 3060).

Key VRAM Optimizations Implemented:
1. FP16 Precision:
   - Weights loaded in torch.float16 (variant="fp16") reduce model memory footprint
     by ~50% compared to standard float32 precision.
2. Sequential Model CPU Offloading:
   - `pipe.enable_model_cpu_offload()` moves idle sub-modules (Text Encoder, UNet, VAE)
     to system RAM (CPU) and streams only the actively executing module to GPU VRAM.
     This avoids holding the entire multi-gigabyte pipeline in VRAM concurrently.
3. Sliced VAE Decoding:
   - `decode_chunk_size` decodes latent frames in small batches rather than decoding
     all video frames simultaneously, dramatically cutting peak VRAM spikes during
     the final video decoding stage.
4. Native Resolution Resizing:
   - Resizing input frames to 1024x576 (or configured SVD_RESOLUTION) prevents
     unnecessary spatial memory scaling inside the attention layers.
"""

import os
import time
from typing import Optional

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image

from config import (
    SVD_DECODE_CHUNK_SIZE,
    SVD_FPS,
    SVD_MODEL_ID,
    SVD_NUM_FRAMES,
    SVD_RESOLUTION,
    SVD_SEED,
    TEMP_DIR,
)

# Global pipeline instance cache to prevent reloading the model on every call
_pipe: Optional[StableVideoDiffusionPipeline] = None


def _load_pipeline() -> StableVideoDiffusionPipeline:
    """Loads and caches the Stable Video Diffusion (SVD-XT) pipeline with VRAM optimizations.

    Uses a global singleton cache `_pipe` so that the model weights are loaded
    into memory only once across multiple calls.

    Optimization Details:
    - `torch_dtype=torch.float16` & `variant='fp16'`: Cuts parameter memory from ~8GB to ~4GB.
    - `pipe.enable_model_cpu_offload()`: Offloads inactive model sub-modules to CPU memory,
      ensuring only the currently executing component occupies GPU VRAM. This is essential
      for operating within 6GB VRAM bounds.

    Returns:
        StableVideoDiffusionPipeline: Configured and cached SVD-XT pipeline.
    """
    global _pipe
    if _pipe is None:
        print("[Video Generator] Loading SVD-XT model (this may take a while on first run)...")
        # Load weights in half precision (fp16) to minimize initial memory footprint
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        print("[Video Generator] Enabling CPU offload for 6GB VRAM optimization...")
        # Offload pipeline sub-modules to CPU RAM when not in use; streams active components to GPU
        pipe.enable_model_cpu_offload()

        _pipe = pipe

    return _pipe


def generate_bridge_video(
    start_frame_path: str,
    num_frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """Generates a short video starting from the given frame image using SVD-XT.

    Args:
        start_frame_path: Path to the starting frame image.
        num_frames: Number of video frames to generate. Defaults to SVD_NUM_FRAMES if None.
        seed: Random seed for deterministic generation. Defaults to SVD_SEED if None.

    Returns:
        str: File path to the exported MP4 bridge video.

    Raises:
        FileNotFoundError: If start_frame_path does not exist on disk.
        torch.cuda.OutOfMemoryError: If GPU memory limit is exceeded during generation.
        Exception: If any other error occurs during image loading, generation, or export.
    """
    # Apply default parameters from config if not explicitly provided
    if num_frames is None:
        num_frames = SVD_NUM_FRAMES
    if seed is None:
        seed = SVD_SEED

    # Validate start frame path existence
    if not os.path.exists(start_frame_path):
        raise FileNotFoundError(
            f"[Video Generator] Start frame image not found at path: {start_frame_path}"
        )

    print("[Video Generator] Preparing start frame...")
    try:
        # Load image via PIL and ensure RGB format (strips alpha channel if present)
        image = Image.open(start_frame_path).convert("RGB")

        # Resize to standard SVD-XT resolution (e.g. 1024x576) using high-quality LANCZOS resampling.
        # Native dimensions ensure the attention layers stay within predictable memory limits.
        image = image.resize(SVD_RESOLUTION, Image.LANCZOS)
    except Exception as img_err:
        print(f"[Video Generator ERROR] Failed to load or resize start frame: {img_err}")
        raise

    print(f"[Video Generator] Generating {num_frames} frames at {SVD_RESOLUTION}...")
    print("[Video Generator] This will take ~3-7 minutes on RTX 4050. Please be patient...")

    start_time = time.time()

    try:
        # Retrieve or initialize the cached pipeline
        pipe = _load_pipeline()

        # Set manual seed for reproducibility
        generator = torch.manual_seed(seed)

        # Generate video latent frames.
        # decode_chunk_size: Decodes latent frames in smaller sub-batches to prevent VAE OOM spikes.
        # num_inference_steps: 25 steps provides a balanced tradeoff between quality and execution time.
        output = pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=SVD_DECODE_CHUNK_SIZE,
            generator=generator,
            num_inference_steps=25,
            motion_bucket_id=50,       # Lower motion bucket for smoother, more stable movement
            noise_aug_strength=0.05,   # Slightly higher noise aug to prevent rapid degradation
        )
        frames = output.frames[0]

        elapsed = time.time() - start_time
        print(f"[Video Generator] Generation complete in {elapsed:.1f}s ({len(frames)} frames)")

        # Ensure output directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        output_path = os.path.join(TEMP_DIR, "bridge_generated.mp4")

        # Export generated frames list to an MP4 video file
        export_to_video(frames, output_path, fps=SVD_FPS)
        print(f"[Video Generator] Bridge video saved to {output_path}")

        return output_path

    except torch.cuda.OutOfMemoryError as oom_error:
        print(
            "\n[Video Generator ERROR] CUDA Out of Memory (OOM) encountered during video generation!\n"
            "VRAM Optimization Recommendations for 6GB GPUs:\n"
            "  1. Lower `num_frames` (e.g., reduce from 25 to 14 frames in config).\n"
            "  2. Lower `decode_chunk_size` (e.g., set to 2 or 4 to minimize VAE memory usage).\n"
            "  3. Close background GPU-heavy applications (e.g., browser tabs, other AI tools).\n"
            "  4. Ensure CPU offloading is active via `pipe.enable_model_cpu_offload()`."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise oom_error

    except Exception as gen_error:
        print(f"[Video Generator ERROR] Unexpected error during video generation/export: {gen_error}")
        raise
