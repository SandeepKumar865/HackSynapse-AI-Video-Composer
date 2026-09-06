from typing import List, Optional
from pydantic import BaseModel, Field

class CharacterVisual(BaseModel):
    age: int
    gender: str
    body: str
    clothing: str
    appearance: str

class VoiceConfig(BaseModel):
    language: str = "en"
    tone: str
    pitch: str
    speed: str

class Character(BaseModel):
    id: str = Field(..., description="Unique ID for character, e.g., char_1")
    name: str
    visual: CharacterVisual
    voice: VoiceConfig
    reference_asset: Optional[str] = Field(None, description="Path to generated reference image")

class Background(BaseModel):
    id: str = Field(..., description="Unique ID for background, e.g., bg_1")
    description: str
    time_of_day: str
    reference_asset: Optional[str] = Field(None, description="Path to generated reference image")

class Camera(BaseModel):
    shot_type: str = Field(..., description="e.g., wide, medium, close_up")
    movement: str = Field(..., description="e.g., slow_dolly_forward, static, pan_right")
    angle: str = Field(..., description="e.g., eye_level, low_angle, high_angle")

class Animation(BaseModel):
    character_motion: str = Field(..., description="e.g., slow cautious movement")
    camera_motion: str = Field(..., description="e.g., slow_push_in")
    environment_motion: List[str] = Field(..., description="e.g., ['falling rain', 'steam drifting']")
    effects: List[str] = Field(..., description="e.g., ['artifact glow pulse']")

class SFXEvent(BaseModel):
    type: str = Field(..., description="e.g., footsteps, neon_hum")
    timestamp: float = Field(..., description="Time in seconds from the start of the scene")
    duration: float = Field(..., description="Duration of the sound effect")
    volume: float = Field(..., description="Volume level from 0.0 to 1.0")

class SceneMusic(BaseModel):
    intensity: float = Field(..., description="Intensity of the music in this scene from 0.0 to 1.0")
    volume: float = Field(..., description="Volume level from 0.0 to 1.0")

class AudioTimeline(BaseModel):
    sfx: List[SFXEvent] = []
    music: SceneMusic

class Transition(BaseModel):
    type: str = Field(..., description="e.g., cut, fade, dissolve")
    duration: float = Field(0.0, description="Duration of the transition")

class Dialogue(BaseModel):
    speaker_id: str
    text: str

class Narration(BaseModel):
    voice_id: str
    text: str

class Scene(BaseModel):
    id: str = Field(..., description="Scene ID, e.g., scene_01")
    duration: float = Field(..., description="Duration of the scene in seconds (vary this!)")
    background_id: str
    transition_reason: Optional[str] = Field(None, description="Logical story reason for changing backgrounds if different from previous scene")
    character_ids: List[str]
    shot: Camera
    animation: Animation
    action: str = Field(..., description="The narrative action happening in the scene")
    visual_prompt: str = Field(..., description="Rich, descriptive prompt for Image-to-Video generation (comma separated keywords)")
    dialogue: Optional[Dialogue] = Field(None, description="Character speech")
    narration: Optional[Narration] = Field(None, description="Voiceover narration")
    audio: AudioTimeline
    transition: Transition

class ProjectAudioConfig(BaseModel):
    mood: str = Field(..., description="Global mood of the video's music bed")
    track: str = Field(..., description="Specific track ID or description")

class ProjectMetadata(BaseModel):
    title: str
    target_duration: float
    style: str
    aspect_ratio: str = "16:9"
    audio: ProjectAudioConfig

class MasterPlan(BaseModel):
    project: ProjectMetadata
    characters: List[Character] = Field(..., description="List of living characters. NO inanimate objects.")
    backgrounds: List[Background]
    scenes: List[Scene]
