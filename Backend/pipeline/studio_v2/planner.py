import json
import ollama
from schemas import MasterPlan

def normalize_plan(plan: MasterPlan) -> MasterPlan:
    """
    Python Enforcement Layer.
    LLMs are bad at math. This function mathematically scales scene durations 
    so they sum exactly to the target_duration, and clamps audio/transition bounds.
    """
    target = plan.project.target_duration
    
    # 1. Normalize Scene Durations
    current_total = sum(scene.duration for scene in plan.scenes)
    if current_total == 0:
        return plan # Avoid division by zero if model completely hallucinated
        
    scale_factor = target / current_total
    
    new_total = 0.0
    for i, scene in enumerate(plan.scenes):
        if i == len(plan.scenes) - 1:
            # Last scene absorbs any floating point rounding errors to ensure exact match
            scene.duration = round(target - new_total, 1)
        else:
            scene.duration = round(scene.duration * scale_factor, 1)
            new_total += scene.duration
            
    # 2. Clamp Audio and Transition Boundaries
    for scene in plan.scenes:
        # Clamping transitions
        if scene.transition.duration > scene.duration:
            scene.transition.duration = scene.duration
            
        # Clamping SFX
        for sfx in scene.audio.sfx:
            # If SFX starts after the scene ends, pull it back to the start
            if sfx.timestamp >= scene.duration:
                sfx.timestamp = 0.0
            
            # If SFX bleeds over the end of the scene, clamp the duration
            max_duration = scene.duration - sfx.timestamp
            if sfx.duration > max_duration:
                sfx.duration = round(max_duration, 1)
                
    return plan

def generate_master_plan(user_prompt: str, model_name: str = "qwen2.5:3b", chunk_callback = None) -> MasterPlan:
    """
    Takes a user prompt and uses a local LLM via Ollama to generate a highly structured 
    Cinematic MasterPlan JSON matching our Pydantic schema.
    """
    
    target_duration = 15.0
    scene_count = 5
    
    template = '''{
  "project": {
    "title": "Title",
    "target_duration": 15.0,
    "style": "cinematic, photorealistic, 4k",
    "aspect_ratio": "16:9",
    "audio": {
      "mood": "suspenseful",
      "track": "main_theme_01"
    }
  },
  "characters": [
    {
      "id": "char_1",
      "name": "Alex",
      "visual": {
        "age": 25,
        "gender": "male",
        "body": "average",
        "clothing": "casual jacket",
        "appearance": "short hair, determined look"
      },
      "voice": {
        "language": "en",
        "tone": "confident",
        "pitch": "medium",
        "speed": "medium"
      },
      "reference_asset": null
    }
  ],
  "backgrounds": [
    {
      "id": "bg_1",
      "description": "A sunny suburban street",
      "time_of_day": "day",
      "reference_asset": null
    }
  ],
  "scenes": [
    {
      "id": "scene_01",
      "duration": 3.5,
      "background_id": "bg_1",
      "transition_reason": null,
      "character_ids": ["char_1"],
      "shot": {
        "shot_type": "wide establishing",
        "movement": "slow_dolly_forward",
        "angle": "eye_level"
      },
      "animation": {
        "character_motion": "walking",
        "camera_motion": "slow_dolly_forward",
        "environment_motion": ["wind in trees"],
        "effects": []
      },
      "action": "Alex walks down the street.",
      "visual_prompt": "Cinematic wide shot, Alex walking down a sunny street, photorealistic",
      "dialogue": null,
      "narration": null,
      "audio": {
        "sfx": [
          {"type": "footsteps", "timestamp": 0.5, "duration": 2.0, "volume": 0.6}
        ],
        "music": {
          "intensity": 0.3,
          "volume": 0.5
        }
      },
      "transition": {
        "type": "cut",
        "duration": 0.0
      }
    }
  ]
}'''

    system_prompt = (
        "You are an expert AI cinematic director. Your job is to output ONLY a valid JSON object matching the template below.\n"
        "Do not include markdown formatting, backticks, or conversational text. ONLY JSON. DO NOT INCLUDE ANY TRAILING TEXT AFTER THE JSON.\n\n"
        "CINEMATIC DIRECTIVES (CRITICAL):\n"
        "1. ADAPT TO THE USER PROMPT: The story, characters, and style MUST match the user's idea entirely! (Do NOT copy the generic template characters or style if it doesn't match).\n"
        "2. DYNAMIC TIMING: Vary scene durations (e.g., 2.5s, 4.0s, 1.5s) based on action intensity.\n"
        "3. SPARSE DIALOGUE: Use `dialogue: null` mostly. Max 1-2 lines in a trailer. Use silence for tension.\n"
        "4. CAMERA LANGUAGE: Use highly dynamic, varied shot types. Do NOT repeat static shots.\n"
        "5. VISUAL PROMPT: Highly detailed, comma-separated keywords for an Image-to-Video generator. NO FLUFF like '8k'.\n"
        f"6. EXACT SCENE COUNT: You MUST generate EXACTLY {scene_count} scenes. Stop generating after the {scene_count}th scene.\n"
        f"7. TOTAL DURATION: The durations of your {scene_count} scenes MUST mathematically add up to exactly {target_duration} seconds.\n"
        "8. LOCATION CONSISTENCY: Do not jump to unprompted locations. If `background_id` changes from the previous scene, you MUST provide a logical `transition_reason`.\n"
        f"TEMPLATE (USE AS A STRUCTURAL GUIDE ONLY):\n{template}"
    )
    
    print(f"Generating Cinematic Master Plan using {model_name}... (High-Speed Mode)")
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a short video script based on this idea: {user_prompt}"}
        ],
        options={"temperature": 0.0},
        stream=True
    )
    
    json_output = ""
    for chunk in response:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        json_output += content
        if chunk_callback:
            chunk_callback(json_output)
        
    print("\n\nParsing JSON...")
    
    # Very aggressive JSON extraction to fix trailing characters issue
    import re
    import json_repair
    
    match = re.search(r'\{.*\}', json_output, re.DOTALL)
    if match:
        json_output = match.group(0)
        
    print("\nRepairing LLM JSON Syntax...")
    json_output = json_repair.repair_json(json_output)

    try:
        plan = MasterPlan.model_validate_json(json_output)
    except Exception as e:
        print(f"\n[FATAL] JSON Validation Failed! LLM output was invalid. Trying to dump raw output for debug.")
        import os
        os.makedirs("output", exist_ok=True)
        with open("output/failed_llm_response.txt", "w") as f:
            f.write(json_output)
        raise e
    
    if len(plan.scenes) > scene_count:
        print(f"\n[WARNING] Model generated {len(plan.scenes)} scenes. Trimming down to {scene_count}.")
        plan.scenes = plan.scenes[:scene_count]
        
    print("Normalizing durations and audio boundaries...")
    plan = normalize_plan(plan)
        
    return plan

if __name__ == "__main__":
    test_prompt = "A 15-second cinematic trailer about a cyberpunk detective named Kael finding a glowing artifact in a neon-lit alleyway."
    try:
        print(f"Prompt: {test_prompt}\n")
        plan = generate_master_plan(test_prompt)
        
        output_file = "master_plan.json"
        with open(output_file, "w") as f:
            f.write(plan.model_dump_json(indent=4))
            
        print(f"\nSuccess! Cinematic Master Plan saved to {output_file}")
        print("\nGenerated Title:", plan.project.title)
        
        actual_total = sum(s.duration for s in plan.scenes)
        print(f"Total Scenes: {len(plan.scenes)}")
        print(f"Total Duration (Normalized): {actual_total} seconds")
        
    except Exception as e:
        print(f"\nError: {e}")
