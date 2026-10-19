from fastapi import FastAPI

from app.api.video import router as video_router
from app.api.voice_playback import router as voice_playback_router

app = FastAPI(title="Bullseye")

app.include_router(video_router)
app.include_router(voice_playback_router)


@app.get("/")
def root():
    return {
        "project": "Bullseye",
        "status": "running",
    }
