import asyncio
import json
import threading
import torch
import os
from planner import generate_master_plan
from schemas import Scene, MasterPlan

# Suppress HF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

class StreamingOrchestrator:
    def __init__(self):
        self.scene_queue = asyncio.Queue()
        self.processed_scene_ids = set()
        self.loop = None
        self.final_plan = None
        self.characters_cache = None
        self.backgrounds_cache = None
        self.gpu_lock = asyncio.Lock()
        
        # We will lazy-load the pipelines to prevent crashing VRAM on startup
        self.image_pipe = None
        self.video_pipe = None

    def _get_image_pipe(self):
        if self.image_pipe is None:
            from diffusers import AutoPipelineForText2Image
            print("[SYSTEM] Loading SDXL-Turbo into system RAM...")
            self.image_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", 
                torch_dtype=torch.float16, 
                variant="fp16"
            )
            # MAGIC TRICK: Offloads weights to CPU RAM, only pushing to VRAM when computing
            self.image_pipe.enable_model_cpu_offload()
            print("[SYSTEM] SDXL-Turbo loaded successfully.")
        return self.image_pipe

    def _get_video_pipe(self):
        if self.video_pipe is None:
            from diffusers import LTXImageToVideoPipeline
            print("[SYSTEM] Loading LTX-Video into system RAM...")
            self.video_pipe = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video", 
                torch_dtype=torch.bfloat16
            )
            # MAGIC TRICK: VRAM offloading and VAE optimization
            self.video_pipe.enable_model_cpu_offload()
            self.video_pipe.vae.enable_slicing()
            self.video_pipe.vae.enable_tiling()
            print("[SYSTEM] LTX-Video loaded successfully.")
        return self.video_pipe

    def preload_models(self):
        print("\n[SYSTEM] Checking and downloading AI Models if necessary...")
        print("[SYSTEM] (Progress bars will appear below for any missing files)")
        self._get_image_pipe()
        self._get_video_pipe()
        print("[SYSTEM] All models downloaded and cached successfully!\n")

    async def image_worker(self):
        """Worker for Local SDXL Image Generation"""
        while True:
            job = await self.scene_queue.get()
            
            # Handle Pre-generation of Assets (Characters and Backgrounds)
            job_type = job.get("type", "scene")
            if job_type in ["character", "background"]:
                data = job.get("data", {})
                asset_id = data.get("id", "unknown")
                print(f"\n[IMAGE WORKER] Pre-generating {job_type} asset: {asset_id}...")
                
                if job_type == "character":
                    vis = data.get("visual", {})
                    prompt = f"Character concept art, {data.get('name')} ({vis.get('age')}yo {vis.get('gender')}), {vis.get('body')}, {vis.get('clothing')}, {vis.get('appearance')}, cinematic lighting, highly detailed"
                else:
                    prompt = f"Environment concept art, {data.get('visual_description', data.get('description', ''))}, {data.get('time_of_day', '')}, cinematic lighting, highly detailed"
                
                def generate_asset():
                    pipe = self._get_image_pipe()
                    generator = torch.Generator(device="cpu").manual_seed(42)
                    image = pipe(prompt=prompt, num_inference_steps=3, guidance_scale=0.0, width=768, height=512, generator=generator).images[0]
                    os.makedirs(f"output/assets", exist_ok=True)
                    img_path = f"output/assets/{asset_id}.png"
                    image.save(img_path)
                    return img_path
                    
                async with self.gpu_lock:
                    img_path = await asyncio.to_thread(generate_asset)
                print(f"[IMAGE WORKER] Finished {job_type} asset! Saved to {img_path}")
                self.scene_queue.task_done()
                continue
                
            # Otherwise, it's a scene generation job
            scene_dict = job
            scene_id = scene_dict.get("id", "unknown")
            visual_prompt = scene_dict.get("visual_prompt", "")
            
            print(f"\n[IMAGE WORKER] Starting SDXL generation for {scene_id}...")
            
            # Inject Character Profiles into the prompt
            injected_prompt = visual_prompt
            if self.characters_cache:
                for char_id in scene_dict.get("character_ids", []):
                    for char in self.characters_cache:
                        if char.get("id") == char_id:
                            vis = char.get("visual", {})
                            desc = f"{char.get('name')} ({vis.get('age')}yo {vis.get('gender')}, {vis.get('body')}, {vis.get('clothing')}, {vis.get('appearance')})"
                            injected_prompt += f", {desc}"
                            print(f"[IMAGE WORKER] Injected consistent character traits for {char.get('name')}")
            
            # Offload blocking computation to a thread so we don't freeze the async event loop
            def generate_img():
                pipe = self._get_image_pipe()
                # Use a fixed seed for temporal consistency across scenes
                generator = torch.Generator(device="cpu").manual_seed(42)
                # Turbo models only need 1-4 steps! Generate in 768x512 widescreen.
                image = pipe(prompt=injected_prompt, num_inference_steps=3, guidance_scale=0.0, width=768, height=512, generator=generator).images[0]
                scene_dir = f"output/{scene_id}"
                os.makedirs(scene_dir, exist_ok=True)
                img_path = f"{scene_dir}/reference.png"
                image.save(img_path)
                return img_path

            # GPU Lock: Prevent Image and Video workers from running simultaneously!
            async with self.gpu_lock:
                img_path = await asyncio.to_thread(generate_img)
            
            print(f"[IMAGE WORKER] Finished SDXL reference for {scene_id}! Saved to {img_path}")
            
            # Update the dictionary with the generated asset path
            scene_dict["reference_asset"] = img_path
            
            # Pass to video worker
            asyncio.create_task(self.video_worker(scene_dict))
            
            self.scene_queue.task_done()

    async def video_worker(self, scene_dict):
        """Worker for Local LTX-Video Generation"""
        scene_id = scene_dict.get("id", "unknown")
        visual_prompt = scene_dict.get("visual_prompt", "")
        img_path = scene_dict.get("reference_asset")
        
        print(f"[VIDEO WORKER] Starting LTX-Video generation for {scene_id}...")
        
        def generate_vid():
            pipe = self._get_video_pipe()
            from diffusers.utils import load_image
            
            # Load the generated SDXL image as the starting frame
            init_image = load_image(img_path)
            
            # Inject dynamic motion into the prompt for much better video generation
            action = scene_dict.get("action", "")
            anim = scene_dict.get("animation", {})
            shot = scene_dict.get("shot", {})
            motion_desc = f"{action}, {shot.get('movement', '')}, {anim.get('camera_motion', '')}, {anim.get('character_motion', '')}"
            video_prompt = f"{visual_prompt}. {motion_desc}"
            
            # 8GB VRAM HARD LIMIT: We cannot exceed 17 frames at 768x512 without triggering 
            # massive RAM swapping (which causes the 37 minute generation time!).
            duration = scene_dict.get("duration", 2.0)
            num_frames = 17
            
            # Keep resolution strictly at 768x512 for LTX-Video optimal quality
            video = pipe(
                image=init_image,
                prompt=video_prompt,
                width=768,
                height=512,
                num_frames=num_frames, 
                num_inference_steps=20, # Reduced from 25 to safely speed it up
            ).frames[0]
            
            from diffusers.utils import export_to_video
            scene_dir = f"output/{scene_id}"
            os.makedirs(scene_dir, exist_ok=True)
            vid_path = f"{scene_dir}/video.mp4"
            
            # To make it match the scene's exact duration, we dynamically change the framerate!
            # 17 frames / 2.5 seconds = 6.8 FPS (Cinematic slow motion!)
            export_fps = max(4, int(num_frames / duration))
            export_to_video(video, vid_path, fps=export_fps)
            return vid_path

        try:
            # GPU Lock: Prevent Image and Video workers from running simultaneously!
            async with self.gpu_lock:
                vid_path = await asyncio.to_thread(generate_vid)
            
            print(f"[VIDEO WORKER] Finished Video for {scene_id}! Saved to {vid_path}")
        except Exception as e:
            print(f"[VIDEO WORKER ERROR] Failed to generate {scene_id}: {e}")

    def parse_stream(self, current_text: str):
        """Real-time JSON stream parser using brace counting."""
        # 1. Try to capture the characters array for persistent prompting
        if not self.characters_cache:
            import re
            char_match = re.search(r'"characters":\s*(\[\s*\{.*?\}\s*\])\s*,\s*"backgrounds"', current_text, re.DOTALL)
            if char_match:
                try:
                    self.characters_cache = json.loads(char_match.group(1))
                    print(f"\n[SYSTEM] Captured {len(self.characters_cache)} characters. Queueing for pre-generation!")
                    for char in self.characters_cache:
                        self.loop.call_soon_threadsafe(self.scene_queue.put_nowait, {"type": "character", "data": char})
                except Exception as e:
                    pass

        # 2. Try to capture the backgrounds array for persistent prompting
        if not self.backgrounds_cache:
            import re
            bg_match = re.search(r'"backgrounds":\s*(\[\s*\{.*?\}\s*\])\s*,\s*"scenes"', current_text, re.DOTALL)
            if bg_match:
                try:
                    self.backgrounds_cache = json.loads(bg_match.group(1))
                    print(f"\n[SYSTEM] Captured {len(self.backgrounds_cache)} backgrounds. Queueing for pre-generation!")
                    for bg in self.backgrounds_cache:
                        self.loop.call_soon_threadsafe(self.scene_queue.put_nowait, {"type": "background", "data": bg})
                except Exception as e:
                    pass

        # 3. Parse Scenes
        scenes_idx = current_text.find('"scenes": [')
        if scenes_idx == -1:
            return
            
        array_content = current_text[scenes_idx + 11:]
        
        brace_count = 0
        in_string = False
        escape = False
        start_idx = -1
        
        for i, char in enumerate(array_content):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        # Found a complete object!
                        scene_json_str = array_content[start_idx:i+1]
                        try:
                            scene_dict = json.loads(scene_json_str)
                            scene_id = scene_dict.get("id")
                            if scene_id and scene_id not in self.processed_scene_ids:
                                self.processed_scene_ids.add(scene_id)
                                if self.loop:
                                    self.loop.call_soon_threadsafe(self.scene_queue.put_nowait, scene_dict)
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1

    def run_planner_sync(self, prompt: str):
        try:
            self.final_plan = generate_master_plan(prompt, chunk_callback=self.parse_stream)
        except Exception as e:
            print(f"Planner error: {e}")

    async def run(self, prompt: str):
        self.loop = asyncio.get_running_loop()
        
        # Pre-download models synchronously so you can actually see the progress bars
        self.preload_models()
        
        image_task = asyncio.create_task(self.image_worker())
        
        print(f"Starting Streaming Pipeline for prompt: '{prompt}'\n")
        
        planner_thread = threading.Thread(target=self.run_planner_sync, args=(prompt,))
        planner_thread.start()
        
        while planner_thread.is_alive():
            await asyncio.sleep(0.1)
            
        await self.scene_queue.join()
        image_task.cancel()
        
        print("\nAll parallel generation complete!")
        if self.final_plan:
            actual_total = sum(s.duration for s in self.final_plan.scenes)
            print(f"Final Normalized Duration: {actual_total} seconds")

if __name__ == "__main__":
    test_prompt = "A 15-second cinematic trailer about a cyberpunk detective named Kael finding a glowing artifact in a neon-lit alleyway."
    orchestrator = StreamingOrchestrator()
    asyncio.run(orchestrator.run(test_prompt))
