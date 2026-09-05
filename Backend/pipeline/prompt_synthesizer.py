"""
Prompt Synthesizer Module
=========================
This module uses Ollama with the LLaVA vision model to visually analyze
two video frames (start and end frames) and synthesize a cohesive scene
description and transition prompt.

Note:
    Stable Video Diffusion (SVD-XT) does not use text prompts directly
    during generation; however, this visual analysis provides valuable
    scene understanding for the user and ensures forward compatibility
    for future upgrades to text-conditioned video models.
"""

from pathlib import Path
from typing import Optional
import os

import ollama

from config import OLLAMA_MODEL


def synthesize_prompt(
    start_frame_path: str,
    end_frame_path: str,
    user_prompt: str,
) -> str:
    """Send both frame images to LLaVA for visual analysis and prompt synthesis.

    Analyzes the starting and ending frame images using the configured Ollama LLaVA
    vision model. The model assesses scene elements, lighting, camera angles,
    subject motion cues, notes the differences between frames, and incorporates
    the user's creative prompt to generate a smooth video transition description.

    Args:
        start_frame_path: File path to the starting frame image.
        end_frame_path: File path to the ending frame image.
        user_prompt: User-provided creative guidance or transition intent.

    Returns:
        The synthesized scene description and video transition prompt.
        If LLaVA analysis fails or Ollama is unavailable, returns a fallback prompt.

    Raises:
        FileNotFoundError: If start_frame_path or end_frame_path does not exist.
    """
    start_path = Path(start_frame_path)
    end_path = Path(end_frame_path)

    # Verify both image files exist
    if not start_path.is_file():
        raise FileNotFoundError(
            f"[Prompt Synthesizer] Start frame file not found: {start_frame_path}"
        )
    if not end_path.is_file():
        raise FileNotFoundError(
            f"[Prompt Synthesizer] End frame file not found: {end_frame_path}"
        )

    print("[Prompt Synthesizer] Analyzing frames with LLaVA...")

    instruction = (
        "You are an expert AI video director and scene analyzer. "
        "Please analyze the two provided video frames (Frame 1: start frame, Frame 2: end frame) "
        "and perform the following:\n"
        "1. Analyze both video frames (describe the scene, lighting, camera angle, subjects, motion cues).\n"
        "2. Note the differences between the two frames.\n"
        "3. Incorporate the user's creative prompt: "
        f"\"{user_prompt}\"\n"
        "4. Output a single, detailed video generation prompt that describes a smooth transition from frame 1 to frame 2."
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": instruction,
                    "images": [str(start_path.resolve()), str(end_path.resolve())],
                }
            ],
        )

        # Extract response text (supporting ChatResponse object and dict fallback)
        if hasattr(response, "message") and hasattr(response.message, "content"):
            synthesized_prompt = response.message.content
        elif isinstance(response, dict) and "message" in response:
            synthesized_prompt = response["message"]["content"]
        else:
            synthesized_prompt = str(response)

        synthesized_prompt = synthesized_prompt.strip()

        divider = "=" * 60
        print("\n[Prompt Synthesizer] Generated scene description:")
        print(divider)
        print(synthesized_prompt)
        print(divider)

        return synthesized_prompt

    except Exception as exc:
        print(f"[Prompt Synthesizer] Warning: LLaVA analysis failed: {exc}")
        print(f"[Prompt Synthesizer] Error details: {exc}")
        fallback_prompt = f"Smooth video transition. {user_prompt}"
        return fallback_prompt
