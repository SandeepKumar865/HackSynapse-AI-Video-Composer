import os
import json
import argparse
from planner import generate_master_plan

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

class Orchestrator:
    def __init__(self):
        self.scenes = []
        
    def parse_stream(self, chunk: str):
        # We don't need real-time streaming parsing if we just wait for the final JSON
        pass

    def run(self, prompt: str):
        print(f"\n[SYSTEM] Starting Planner for movie prompt: '{prompt}'")
        
        try:
            final_plan = generate_master_plan(prompt, chunk_callback=self.parse_stream)
        except Exception as e:
            print(f"[FATAL] Planner error: {e}")
            return
            
        print(f"\n[SYSTEM] Planning complete. Total scenes: {len(final_plan.scenes)}")
        
        os.makedirs("output", exist_ok=True)
        
        # Save master plan
        plan_dict = final_plan.model_dump()
        with open("output/master_plan.json", "w", encoding="utf-8") as f:
            json.dump(plan_dict, f, indent=4)
            
        # Extract and save scenes
        scenes_export = []
        for scene in plan_dict.get("scenes", []):
            scene_id = scene.get("id")
            
            # Combine characters into visual prompt
            visual_prompt = scene.get("visual_prompt", "")
            if "characters" in plan_dict:
                for char_id in scene.get("character_ids", []):
                    for char in plan_dict["characters"]:
                        if char.get("id") == char_id:
                            vis = char.get("visual_description", "")
                            visual_prompt += f", {vis}"
                            
            scenes_export.append({
                "id": scene_id,
                "duration": scene.get("duration", 2.0),
                "action": scene.get("action", ""),
                "visual": visual_prompt
            })
            os.makedirs(f"output/{scene_id}", exist_ok=True)
            
        with open("output/scenes.json", "w", encoding="utf-8") as f:
            json.dump(scenes_export, f, indent=4)
            
        print("[SUCCESS] scenes.json generated successfully. Ready for video generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Movie prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("       AI MOVIE ORCHESTRATOR")
    print("=" * 70)
    
    user_prompt = args.prompt
    if not user_prompt:
        user_prompt = input("\nEnter your movie idea: ").strip()
        
    if not user_prompt:
        user_prompt = "A 15-second cinematic trailer about a cyberpunk detective named Kael finding a glowing artifact in a neon-lit alleyway."
        print(f"Using default prompt: {user_prompt}")
        
    orchestrator = Orchestrator()
    orchestrator.run(user_prompt)
