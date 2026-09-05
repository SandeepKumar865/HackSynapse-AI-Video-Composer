"""
video_stitcher.py — Video Slicing, Speed Adjustment & Seamless Concatenation

This module handles the final assembly of the inpainted video:
1. Slices the original video into two parts (before and after the removed segment)
2. Loads the AI-generated bridge video
3. Adjusts the bridge speed to exactly fill the time gap
4. Applies cross-fade blending at the transition boundaries
5. Concatenates everything into a seamless final output

Uses MoviePy 2.0 API (subclipped instead of deprecated subclip).
"""

import os
from moviepy import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)


def stitch_video(
    source_path: str,
    start_time: float,
    end_time: float,
    bridge_path: str,
    output_path: str,
    crossfade_duration: float = 0.5,
) -> str:
    """
    Stitch together the original video with the AI-generated bridge segment.

    Takes the original video, removes the segment between start_time and end_time,
    and replaces it with the bridge video (speed-adjusted to fit the gap exactly).
    A short cross-fade is applied at transition points for visual smoothness.

    Args:
        source_path: Path to the original source video file.
        start_time: Start time (seconds) of the segment that was removed.
        end_time: End time (seconds) of the segment that was removed.
        bridge_path: Path to the AI-generated bridge video from SVD-XT.
        output_path: Path where the final stitched video will be saved.
        crossfade_duration: Duration (seconds) of cross-fade at transitions.
                           Default 0.5s. Set to 0 to disable.

    Returns:
        The absolute path to the final output video file.

    Raises:
        FileNotFoundError: If source_path or bridge_path don't exist.
        ValueError: If timing parameters are invalid.
        RuntimeError: If video processing fails.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source video not found: {source_path}")
    if not os.path.isfile(bridge_path):
        raise FileNotFoundError(f"Bridge video not found: {bridge_path}")
    if start_time >= end_time:
        raise ValueError(
            f"start_time ({start_time}) must be less than end_time ({end_time})"
        )

    # Track all clips for cleanup
    clips_to_close = []

    try:
        # ------------------------------------------------------------------
        # Step 1: Load and slice the original video
        # ------------------------------------------------------------------
        print("[Stitcher] Loading source video...")
        source = VideoFileClip(source_path)
        clips_to_close.append(source)

        source_duration = source.duration
        if end_time > source_duration:
            raise ValueError(
                f"end_time ({end_time}s) exceeds video duration ({source_duration:.2f}s)"
            )

        # Part 1: Everything before the cut point
        # Using subclipped() — the MoviePy 2.0 replacement for subclip()
        print(f"[Stitcher] Slicing Part 1: 0s -> {start_time}s")
        part1 = source.subclipped(0, start_time)
        clips_to_close.append(part1)

        # Part 2: Everything after the cut point
        print(f"[Stitcher] Slicing Part 2: {end_time}s -> {source_duration:.2f}s")
        part2 = source.subclipped(end_time)
        clips_to_close.append(part2)

        # ------------------------------------------------------------------
        # Step 2: Load and speed-adjust the bridge video
        # ------------------------------------------------------------------
        print("[Stitcher] Loading AI-generated bridge video...")
        bridge = VideoFileClip(bridge_path)
        clips_to_close.append(bridge)

        gap_duration = end_time - start_time
        original_bridge_duration = bridge.duration

        print(f"[Stitcher] Bridge duration: {original_bridge_duration:.2f}s")
        print(f"[Stitcher] Gap to fill: {gap_duration:.2f}s")

        # Calculate speed factor to make bridge duration match the gap exactly
        # speed_factor > 1 = speed up, < 1 = slow down
        if original_bridge_duration > 0 and gap_duration > 0:
            speed_factor = original_bridge_duration / gap_duration
            print(f"[Stitcher] Applying speed factor: {speed_factor:.3f}x")

            # Apply speed adjustment using MoviePy 2.0 API
            # with_speed_scaled multiplies the playback speed
            bridge_adjusted = bridge.with_speed_scaled(speed_factor)
            clips_to_close.append(bridge_adjusted)
        else:
            print("[Stitcher] Warning: Invalid durations, using bridge as-is")
            bridge_adjusted = bridge

        # ------------------------------------------------------------------
        # Step 3: Match resolution
        # ------------------------------------------------------------------
        source_size = source.size  # (width, height)
        bridge_size = bridge_adjusted.size

        if source_size != bridge_size:
            print(
                f"[Stitcher] Resizing bridge from {bridge_size} to {source_size}"
            )
            bridge_adjusted = bridge_adjusted.resized(source_size)
            clips_to_close.append(bridge_adjusted)

        # ------------------------------------------------------------------
        # Step 4: Remove audio from bridge (it has no meaningful audio)
        # ------------------------------------------------------------------
        bridge_adjusted = bridge_adjusted.without_audio()

        # ------------------------------------------------------------------
        # Step 5: Apply cross-fade blending at transition boundaries
        # ------------------------------------------------------------------
        # Cross-fade creates a smooth visual transition:
        # - End of Part 1 blends into start of bridge
        # - End of bridge blends into start of Part 2
        # This compensates for SVD only conditioning on the start frame
        if crossfade_duration > 0 and crossfade_duration < gap_duration / 2:
            print(
                f"[Stitcher] Applying {crossfade_duration}s cross-fade at transitions"
            )
            # MoviePy's concatenate_videoclips with crossfadein/out
            # Apply crossfade by using the transition parameter
            bridge_adjusted = bridge_adjusted.crossfadein(crossfade_duration)
            bridge_adjusted = bridge_adjusted.crossfadeout(crossfade_duration)
            clips_to_close.append(bridge_adjusted)

            # Use compositing method for proper cross-fade rendering
            final = concatenate_videoclips(
                [part1, bridge_adjusted, part2],
                method="compose",
                padding=-crossfade_duration,
            )
        else:
            # Simple hard-cut concatenation (no cross-fade)
            print("[Stitcher] Using hard-cut concatenation (no cross-fade)")
            final = concatenate_videoclips(
                [part1, bridge_adjusted, part2],
                method="compose",
            )

        clips_to_close.append(final)

        # ------------------------------------------------------------------
        # Step 6: Write the final output
        # ------------------------------------------------------------------
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"[Stitcher] Writing final video to: {output_path}")
        print("[Stitcher] Encoding with libx264 (this may take a moment)...")

        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            # Use reasonable encoding settings
            preset="medium",
            # Suppress excessive ffmpeg output
            logger=None,
        )

        # Verify output was created
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Failed to create output file: {output_path}")

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Stitcher] OK Final video saved: {output_path}")
        print(f"[Stitcher]   Size: {output_size_mb:.1f} MB")
        print(f"[Stitcher]   Duration: {final.duration:.2f}s")

        return os.path.abspath(output_path)

    except Exception as e:
        raise RuntimeError(f"Video stitching failed: {e}") from e

    finally:
        # ------------------------------------------------------------------
        # Cleanup: Close all clips to release file handles and memory
        # ------------------------------------------------------------------
        print("[Stitcher] Cleaning up video resources...")
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception:
                pass  # Best-effort cleanup
