from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router
from config import Config

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/keyframes", StaticFiles(directory=Config.KEYFRAMES_DIR),
          name="keyframes")
app.mount("/videos", StaticFiles(directory=Config.VIDEOS_DIR), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
