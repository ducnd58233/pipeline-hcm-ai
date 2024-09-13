from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes import search, grid, frame, panel
from config import Config
from app.log import logger

logger = logger.getChild(__name__)
templates = Jinja2Templates(directory="app/templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/keyframes", StaticFiles(directory=Config.KEYFRAMES_DIR),
          name="keyframes")
# app.mount("/videos", StaticFiles(directory=Config.VIDEOS_DIR), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(search.router)
app.include_router(grid.router)
app.include_router(frame.router)
app.include_router(panel.router)
