from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.services.faiss_service import faiss_service
from app.services.csv_service import save_to_csv
from app.models import FrameMetadata
import logging

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), page: int = Form(1)):
    logger.info(f"Searching for query: {query}, page: {page}")
    results = await faiss_service.search(query, page=page, per_page=20, extra_k=50)
    return templates.TemplateResponse("components/search_results.html", {"request": request, "results": results, "page": page, "query": query})


@router.post("/select_frame", response_class=HTMLResponse)
async def select_frame(request: Request, frame_id: str = Form(...)):
    logger.info(f"Selecting frame: {frame_id}")
    frame = await FrameMetadata.get_by_id(frame_id)
    if frame:
        return templates.TemplateResponse("components/frame_card.html", {"request": request, "frame": frame.to_dict(), "selected": True, "in_search_results": False})
    return HTMLResponse(status_code=404)


@router.post("/deselect_frame")
async def deselect_frame(frame_id: str = Form(...)):
    logger.info(f"Deselecting frame: {frame_id}")
    return HTMLResponse(status_code=204)


@router.get("/get_selected_frames", response_class=HTMLResponse)
async def get_selected_frames(request: Request, page: int = 1, frame_ids: list[str] = []):
    frames_per_page = 10
    start = (page - 1) * frames_per_page
    end = start + frames_per_page
    selected_frames = [await FrameMetadata.get_by_id(frame_id) for frame_id in frame_ids[start:end]]
    selected_frames = [frame.to_dict() for frame in selected_frames if frame]
    return templates.TemplateResponse("components/selected_frames.html", {"request": request, "frames": selected_frames, "page": page, "total": len(frame_ids)})


@router.post("/submit")
async def submit(selected_frames: list[str] = Form(...)):
    logger.info(f"Submitting frames: {selected_frames}")
    await save_to_csv(selected_frames)
    return {"message": "Frames submitted successfully"}
