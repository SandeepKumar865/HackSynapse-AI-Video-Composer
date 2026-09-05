"""
main.py — AI Video Inpainting & Keyframe Interpolation Pipeline

CLI orchestrator that ties together all modules:
1. Extracts boundary frames from the source video (OpenCV)
2. Analyzes frames with a local vision LLM (Ollama + LLaVA)
3. Generates an AI bridge video (Stable Video Diffusion SVD-XT)
4. Stitches everything into a seamless final output (MoviePy)

Usage:
    python main.py --video input.mp4 --start 5.0 --end 10.0 \\
                   --prompt "A magical portal opens" --output final.mp4

All processing runs locally on your machine. No paid APIs required.
Optimized for NVIDIA RTX 4050 (6GB VRAM) + 24GB RAM.
"""

import argparse
import os
import shutil
import sys
import time

# Local module imports
from config import TEMP_DIR, SVD_NUM_FRAMES, SVD_SEED
from frame_extractor import extract_frame, get_video_info
from prompt_synthesizer import synthesize_prompt
from video_generator import generate_bridge_video
from video_stitcher import stitch_video


# ======================================================================
# CLI Argument Parsing
# ======================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="AI Video Inpainter",
        description=(
            "Remove a segment from a video and replace it with an "
            "AI-generated bridge using Stable Video Diffusion.\n"
            "Runs 100%% locally — no paid APIs required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --video clip.mp4 --start 3.0 --end 8.0 "
            '--prompt "Time speeds up"\n'
            "  python main.py --video movie.mp4 --start 10.5 --end 15.0 "
            '--prompt "Scene transitions to night" --output result.mp4\n'
        ),
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the source video file",
    )
    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help="Start time (seconds) of the segment to replace",
    )
    parser.add_argument(
        "--end",
        type=float,
        required=True,
        help="End time (seconds) of the segment to replace",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Creative text prompt describing the desired bridge content",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_final.mp4",
        help="Output file path (default: output_final.mp4)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep intermediate temporary files (frames, bridge video)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SVD_SEED,
        help=f"Random seed for reproducible generation (default: {SVD_SEED})",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=SVD_NUM_FRAMES,
        help=f"Number of frames for SVD to generate (default: {SVD_NUM_FRAMES})",
    )
    parser.add_argument(
        "--no-crossfade",
        action="store_true",
        default=False,
        help="Disable cross-fade blending at transition points",
    )

    return parser.parse_args()


# ======================================================================
# Input Validation
# ======================================================================

def validate_inputs(args: argparse.Namespace) -> dict:
    """
    Validate all inputs before starting the pipeline.

    Returns:
        A dict with video metadata (fps, duration, etc.)

    Raises:
        SystemExit: If validation fails (prints user-friendly error).
    """
    # Check video file exists
    if not os.path.isfile(args.video):
        print(f"\nERROR: Error: Video file not found: {args.video}")
        print("  Please check the file path and try again.")
        sys.exit(1)

    # Check timing values are positive
    if args.start < 0:
        print(f"\nERROR: Error: --start must be non-negative (got {args.start})")
        sys.exit(1)

    if args.end < 0:
        print(f"\nERROR: Error: --end must be non-negative (got {args.end})")
        sys.exit(1)

    # Check start < end
    if args.start >= args.end:
        print(
            f"\nERROR: Error: --start ({args.start}s) must be less than "
            f"--end ({args.end}s)"
        )
        sys.exit(1)

    # Get video info and validate timing against actual duration
    try:
        info = get_video_info(args.video)
    except Exception as e:
        print(f"\nERROR: Error: Could not read video file: {e}")
        sys.exit(1)

    duration = info["duration"]

    if args.start >= duration:
        print(
            f"\nERROR: Error: --start ({args.start}s) is beyond the video "
            f"duration ({duration:.2f}s)"
        )
        sys.exit(1)

    if args.end > duration:
        print(
            f"\nERROR: Error: --end ({args.end}s) is beyond the video "
            f"duration ({duration:.2f}s)"
        )
        sys.exit(1)

    # Warn if the gap is very large (SVD generates ~3.5s of video)
    gap = args.end - args.start
    if gap > 15.0:
        print(
            f"\nWARNING: Warning: The gap ({gap:.1f}s) is much larger than SVD's "
            f"native output (~3.5s). The bridge video will be significantly "
            f"slowed down, which may reduce quality."
        )
        print("  Consider using a smaller gap for best results.\n")

    return info


# ======================================================================
# Main Pipeline
# ======================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the full AI video inpainting pipeline.

    Steps:
        1. Extract boundary frames
        2. Analyze with LLaVA (scene understanding)
        3. Generate bridge video with SVD-XT
        4. Stitch everything together
    """
    pipeline_start = time.time()
    gap_duration = args.end - args.start

    print("=" * 60)
    print("  AI VIDEO INPAINTING PIPELINE")
    print("  100% Local — No Paid APIs")
    print("=" * 60)
    print(f"  Source:   {args.video}")
    print(f"  Replace:  {args.start}s -> {args.end}s ({gap_duration:.2f}s gap)")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Output:   {args.output}")
    print(f"  Seed:     {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Extract boundary frames
    # ------------------------------------------------------------------
    print("\nSTEP 1/4: Extracting boundary frames...")
    step_start = time.time()

    try:
        start_frame_path = extract_frame(args.video, args.start)
        end_frame_path = extract_frame(args.video, args.end)
    except Exception as e:
        print(f"\nERROR: Frame extraction failed: {e}")
        sys.exit(1)

    print(f"   OK Done in {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Analyze frames with LLaVA
    # ------------------------------------------------------------------
    print("\nSTEP 2/4: Analyzing frames with LLaVA...")
    step_start = time.time()

    try:
        scene_description = synthesize_prompt(
            start_frame_path, end_frame_path, args.prompt
        )
    except Exception as e:
        print(f"\nWARNING: LLaVA analysis failed (non-fatal): {e}")
        scene_description = f"Smooth video transition. {args.prompt}"

    print(f"   OK Done in {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Generate bridge video with SVD-XT
    # ------------------------------------------------------------------
    print("\nSTEP 3/4: Generating bridge video with SVD-XT...")
    print("   (This is the slowest step — typically 3-7 minutes)")
    step_start = time.time()

    try:
        bridge_path = generate_bridge_video(
            start_frame_path,
            num_frames=args.num_frames,
            seed=args.seed,
        )
    except Exception as e:
        # Check if it's a CUDA OOM error (torch is lazily imported)
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print("\nERROR: GPU ran out of memory!")
            print("  Try closing other GPU-using applications (games, browsers, etc.)")
            print("  Or reduce --num-frames (e.g., --num-frames 14)")
        else:
            print(f"\nERROR: Video generation failed: {e}")
        sys.exit(1)

    print(f"   OK Done in {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Stitch the final video
    # ------------------------------------------------------------------
    print("\nSTEP 4/4: Stitching final video...")
    step_start = time.time()

    crossfade = 0.0 if args.no_crossfade else 0.5

    try:
        final_path = stitch_video(
            source_path=args.video,
            start_time=args.start,
            end_time=args.end,
            bridge_path=bridge_path,
            output_path=args.output,
            crossfade_duration=crossfade,
        )
    except Exception as e:
        print(f"\nERROR: Video stitching failed: {e}")
        sys.exit(1)

    print(f"   OK Done in {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Cleanup & Summary
    # ------------------------------------------------------------------
    if not args.keep_temp:
        print("\nCleaning up temporary files...")
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            os.makedirs(TEMP_DIR, exist_ok=True)  # Recreate empty dir
            print("   OK Temp files removed")
        except Exception:
            print("   WARNING: Could not fully clean temp directory")
    else:
        print(f"\n📁 Temp files kept in: {TEMP_DIR}")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  OK PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Output file:  {final_path}")
    print(f"  Total time:   {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"  Gap filled:   {gap_duration:.2f}s")
    print("=" * 60)
    print()


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_arguments()

    # Validate inputs before doing any heavy work
    video_info = validate_inputs(args)

    print(f"\nVideo info: {video_info['width']}x{video_info['height']} @ "
          f"{video_info['fps']:.1f}fps, {video_info['duration']:.2f}s total")

    # Import torch here (after validation) to avoid slow import
    # if the user just has a typo in their arguments
    try:
        import torch
    except ImportError:
        print("\nERROR: PyTorch not installed! Install with CUDA support:")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\nWARNING: Warning: CUDA not available. SVD-XT will run on CPU.")
        print("  This will be EXTREMELY slow (potentially hours).")
        print("  Make sure you have NVIDIA drivers and CUDA toolkit installed.")
        response = input("  Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("  Aborted.")
            sys.exit(0)

    # Run the pipeline
    run_pipeline(args)
