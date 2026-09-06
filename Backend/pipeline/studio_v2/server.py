import asyncio
import os
import subprocess
import uuid
import sys
import json
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output", exist_ok=True)
app.mount("/videos", StaticFiles(directory="output"), name="videos")

# In-memory store for job logs and status
jobs = {}

class GenerateRequest(BaseModel):
    prompt: str

def run_pipeline_task(job_id: str, prompt: str):
    jobs[job_id]["status"] = "generating_plan"
    jobs[job_id]["progress"] = 10
    
    # 1. Run Orchestrator
    try:
        process = subprocess.Popen(
            [sys.executable, "orchestrator_v2.py", "--prompt", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            if jobs[job_id]["progress"] < 40:
                jobs[job_id]["progress"] += 0.5  # Increment gradually up to 40
            msg = line.strip()
            if msg:
                jobs[job_id]["logs"].append({"agent": "Planner", "message": msg})
        process.wait()
        
        if process.returncode != 0:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["logs"].append({"agent": "System", "message": f"Orchestrator failed with code {process.returncode}"})
            return
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["logs"].append({"agent": "System", "message": f"Error running orchestrator: {str(e)}"})
        return
        
    jobs[job_id]["status"] = "generating_video"
    jobs[job_id]["progress"] = 75
    
    # 2. Run Generator
    try:
        process = subprocess.Popen(
            [sys.executable, "generate_videos.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            if jobs[job_id]["progress"] < 99:
                jobs[job_id]["progress"] += 0.5
                
            msg = line.strip()
            if msg:
                jobs[job_id]["logs"].append({"agent": "Generator", "message": msg})
                if "[SUCCESS]" in msg and "completed" in msg:
                    jobs[job_id]["completed_scenes"] = jobs[job_id].get("completed_scenes", 0) + 1
        process.wait()
        
        if process.returncode != 0:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["logs"].append({"agent": "System", "message": "Video generation failed"})
            return
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["logs"].append({"agent": "System", "message": str(e)})
        return
        
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 100
    jobs[job_id]["completed_scenes"] = jobs[job_id].get("completed_scenes", 0)
    jobs[job_id]["logs"].append({"agent": "System", "message": "Pipeline finished successfully!"})

@app.post("/api/generate")
async def generate_video(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "starting",
        "progress": 0,
        "logs": [],
        "completed_scenes": 0,
        "prompt": req.prompt
    }
    background_tasks.add_task(run_pipeline_task, job_id, req.prompt)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str, request: Request):
    async def event_generator():
        last_log_index = 0
        while True:
            if await request.is_disconnected():
                break
                
            if job_id not in jobs:
                yield {"data": json.dumps({"error": "Job not found"})}
                break
                
            job = jobs[job_id]
            
            # Send any new logs
            new_logs = job["logs"][last_log_index:]
            last_log_index = len(job["logs"])
            
            scene_count = 0
            try:
                if os.path.exists("output/scenes.json"):
                    with open("output/scenes.json", "r") as f:
                        scenes = json.load(f)
                        scene_count = len(scenes)
            except:
                pass
            
            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "new_logs": new_logs,
                "scene_count": scene_count,
                "completed_scenes": job.get("completed_scenes", 0),
                "video_url": "http://localhost:8000/videos/master_trailer.mp4" if job["status"] == "completed" else None
            }
            yield {"data": json.dumps(payload)}
            
            if job["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
            
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
