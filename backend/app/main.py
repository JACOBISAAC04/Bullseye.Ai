from fastapi import FastAPI

from app.api.video import router as video_router

app = FastAPI(title="Bullseye")

app.include_router(video_router)


@app.get("/")
def root():
    return {
        "project": "Bullseye",
        "status": "running"
    }