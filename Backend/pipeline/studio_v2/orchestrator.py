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
            from diffusers import LTXVideoPipeline
            print("[SYSTEM] Loading LTX-Video into system RAM...")
            self.video_pipe = LTXVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video", 
                torch_dtype=torch.bfloat16
            )
            # MAGIC TRICK: VRAM offloading
            self.video_pipe.enable_model_cpu_offload()
            print("[SYSTEM] LTX-Video loaded successfully.")
        return self.video_pipe

    async def image_worker(self):
        """Worker for Local SDXL Image Generation"""
        while True:
            scene_dict = await self.scene_queue.get()
            scene_id = scene_dict.get("id", "unknown")
            visual_prompt = scene_dict.get("visual_prompt", "")
            
            print(f"\n[IMAGE WORKER] Starting SDXL generation for {scene_id}...")
            
            # Offload blocking computation to a thread so we don't freeze the async event loop
            def generate_img():
                pipe = self._get_image_pipe()
                # Turbo models only need 1-4 steps! Massive speedup.
                image = pipe(prompt=visual_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
                os.makedirs("output", exist_ok=True)
                img_path = f"output/{scene_id}_ref.png"
                image.save(img_path)
                return img_path

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
            
            # Keep resolution extremely small and frame count low for 8GB VRAM constraint
            video = pipe(
                image=init_image,
                prompt=visual_prompt,
                width=512,
                height=320,
                num_frames=33, # roughly 1.3 seconds at 24fps
                num_inference_steps=20,
            ).frames[0]
            
            from diffusers.utils import export_to_video
            vid_path = f"output/{scene_id}_vid.mp4"
            export_to_video(video, vid_path, fps=24)
            return vid_path

        try:
            vid_path = await asyncio.to_thread(generate_vid)
            print(f"[VIDEO WORKER] Finished Video for {scene_id}! Saved to {vid_path}")
        except Exception as e:
            print(f"[VIDEO WORKER ERROR] Failed to generate {scene_id}: {e}")

    def parse_stream(self, current_text: str):
        """Real-time JSON stream parser using brace counting."""
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
