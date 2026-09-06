import os
import json
import subprocess
import traceback
import time

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, logging

logging.disable_progress_bar()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

VIDEO_FRAMES = 24
VIDEO_FPS = 8
INFERENCE_STEPS = 50

SEED = 42

# ============================================================
# HELPERS
# ============================================================

def run_ffmpeg(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print("[FFMPEG ERROR]")
        print(result.stderr)
    return result.returncode == 0

def trim_and_interpolate_video(input_path, output_path, duration):
    """
    Trim generated clip to exact scene duration and interpolate to 24fps.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", str(duration),
        "-filter:v", "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=24'",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    return run_ffmpeg(cmd)

def load_pipeline():
    print(f"[SYSTEM] Loading {MODEL_ID} (float16)...")
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    )

    # VRAM Optimizations for 6GB Cards
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()

    print(f"[SYSTEM] {MODEL_ID} loaded successfully.")
    return pipe

# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("       TEXT-TO-VIDEO GENERATOR (DAMO-VILAB)")
    print("=" * 70)
    print()

    scenes_path = "output/scenes.json"

    try:
        with open(scenes_path, "r", encoding="utf-8") as f:
            scenes = json.load(f)
    except FileNotFoundError:
        print("[ERROR] output/scenes.json not found!")
        return

    if not scenes:
        print("[SYSTEM] No scenes found.")
        return

    os.makedirs("output", exist_ok=True)

    try:
        pipe = load_pipeline()
    except Exception:
        print(f"[FATAL] Failed to load {MODEL_ID} Pipeline.")
        traceback.print_exc()
        return

    processed_scene_ids = []

    for index, scene in enumerate(scenes):
        scene_id = scene.get("id")
        if not scene_id:
            scene_id = f"scene_{index + 1}"
        duration = float(scene.get("duration", 2.0))
        action = scene.get("action", "")
        visual_prompt = scene.get("visual", "")
        
        scene_dir = os.path.join("output", scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # Merge visual prompt with the action, banning cartoon/blurry textures
        prompt = f"{visual_prompt}, {action}, hyperrealistic, cinematic lighting, highly detailed, 4k resolution"
        negative_prompt = "cartoon, 3d, animation, blurry, deformed, distorted, low quality, worst quality, mutated, text, watermark"

        print()
        print("-" * 70)
        print(f"[VIDEO] {scene_id}")
        print(f"[VIDEO] Prompt: {prompt}")
        print(f"[VIDEO] Target duration: {duration:.2f}s")
        print("-" * 70)

        raw_video = os.path.join(scene_dir, "video_raw.mp4")
        final_video = os.path.join(scene_dir, "video.mp4")

        start_time = time.perf_counter()

        try:
            print(f"[DAMO-VILAB] Generating {VIDEO_FRAMES} frames...")
            generator = torch.Generator(device="cpu").manual_seed(SEED + index)

            def progress_callback(step, timestep, latents):
                print(f"[DAMO-VILAB] Step {step}/{INFERENCE_STEPS} completed", flush=True)

            pipe.set_progress_bar_config(disable=True)

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=VIDEO_FRAMES,
                    num_inference_steps=INFERENCE_STEPS,
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1,
                )

            frames = result.frames[0]
            generation_time = time.perf_counter() - start_time
            print(f"[VIDEO] Generation finished in {generation_time:.2f}s")

            print(f"[VIDEO] Exporting {VIDEO_FRAMES} frames @ {VIDEO_FPS} FPS...")
            export_to_video(frames, raw_video, fps=VIDEO_FPS)

            print(f"[VIDEO] Interpolating to 24fps and trimming to {duration:.2f}s...")
            success = trim_and_interpolate_video(raw_video, final_video, duration)

            if not success:
                print(f"[ERROR] FFmpeg failed for {scene_id}")
                continue

            try:
                os.remove(raw_video)
            except OSError:
                pass

            processed_scene_ids.append(scene_id)
            total_time = time.perf_counter() - start_time
            print(f"[SUCCESS] {scene_id} completed in {total_time:.2f}s")
            print(f"[OUTPUT] {final_video}")

        except Exception:
            print()
            print(f"[ERROR] Failed generating {scene_id}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    if not processed_scene_ids:
        print()
        print("[SYSTEM] No videos were successfully generated.")
        return

    print()
    print("=" * 70)
    print("COMPILING FINAL VIDEO")
    print("=" * 70)

    list_file = "output/concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for scene in scenes:
            scene_id = scene.get("id")
            if scene_id not in processed_scene_ids:
                continue
            f.write(f"file '{scene_id}/video.mp4'\n")

    final_video_path = "output/master_trailer.mp4"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", final_video_path
    ]

    success = run_ffmpeg(cmd)

    if success:
        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Master video: {final_video_path}")
    else:
        print("[ERROR] Final video compilation failed.")

if __name__ == "__main__":
    main()

