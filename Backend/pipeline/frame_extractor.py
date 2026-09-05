"""Frame Extractor Module.

This module extracts specific frames from a video file at given timestamps
using OpenCV and retrieves metadata/information about video streams.
"""

import os
from typing import Any, Dict
import cv2

from config import TEMP_DIR


def extract_frame(video_path: str, timestamp: float) -> str:
    """Extract a single frame from a video file at a given timestamp.

    Args:
        video_path (str): The absolute or relative path to the video file.
        timestamp (float): The timestamp in seconds from which to extract the frame.

    Returns:
        str: The file path where the extracted frame JPEG image is saved.

    Raises:
        FileNotFoundError: If the specified video file does not exist.
        RuntimeError: If OpenCV fails to open the video file or read the target frame.
        ValueError: If the calculated target frame is out of the video's frame range.
    """
    # Verify that the video file exists on disk
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    # Open the video stream using OpenCV VideoCapture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file with OpenCV: {video_path}")

    try:
        # Retrieve video properties: frame rate (FPS) and total frame count
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(
                f"Invalid or undetectable frame rate (FPS={fps}) for video: {video_path}"
            )

        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the frame index corresponding to the given timestamp
        frame_num: int = int(timestamp * fps)

        # Validate that the target frame number falls within the allowable frame index range [0, total_frames - 1]
        if frame_num < 0 or frame_num >= total_frames:
            raise ValueError(
                f"Target frame {frame_num} for timestamp {timestamp}s is out of range. "
                f"Total frames available: {total_frames} (0 to {total_frames - 1})."
            )

        # Seek to the target frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame from the video stream
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to read frame at index {frame_num} (timestamp {timestamp}s) from: {video_path}"
            )

        # Ensure the destination directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Construct the target output path for the saved image
        output_path: str = os.path.join(TEMP_DIR, f"frame_{timestamp}s.jpg")

        # Save the extracted frame as a JPEG image
        success = cv2.imwrite(output_path, frame)
        if not success:
            raise RuntimeError(f"Failed to write extracted frame to {output_path}")

        print(f"[Frame Extractor] Saved frame at {timestamp}s -> {output_path}")
        return output_path

    finally:
        # Always release the video capture resource
        cap.release()


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Retrieve metadata and properties of a video file.

    Extracts FPS, total frames, calculated duration, width, and height of the video.

    Args:
        video_path (str): The absolute or relative path to the video file.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'fps' (float): Frames per second of the video.
            - 'total_frames' (int): Total number of frames in the video.
            - 'duration' (float): Total duration in seconds.
            - 'width' (int): Frame width in pixels.
            - 'height' (int): Frame height in pixels.

    Raises:
        FileNotFoundError: If the specified video file does not exist.
        RuntimeError: If OpenCV fails to open the video file.
    """
    # Verify that the video file exists on disk
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    # Open the video stream using OpenCV VideoCapture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file with OpenCV: {video_path}")

    try:
        # Extract video properties
        fps: float = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate duration in seconds
        duration: float = float(total_frames / fps) if fps > 0 else 0.0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
        }

    finally:
        # Always release the video capture resource
        cap.release()
