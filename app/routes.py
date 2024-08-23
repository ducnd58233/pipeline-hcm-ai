from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.services.faiss_service import faiss_service
from app.services.redis_service import redis_service
from app.services.csv_service import save_to_csv
from app.models import FrameMetadata
from app.log import logging_config
import json
import logging

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), page: int = Form(1)):
    try:
        logger.info(f"Searching for query: {query}, page: {page}")
        results = await faiss_service.search(query, page=page, per_page=20, extra_k=50)
        return templates.TemplateResponse("components/search_results.html", {"request": request, "results": results, "page": page, "query": query})
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while searching. Please try again later.")


@router.post("/toggle_frame", response_class=HTMLResponse)
async def toggle_frame(
    request: Request,
    frame_id: str = Form(...),
    score: float = Form(...),
    selected: bool = Form(False),
):
    try:
        logger.info(
            f"Toggling frame ID: {frame_id}, Score: {score}, Selected: {selected}")

        frame = await FrameMetadata.get_by_frame_id(frame_id)
        if not frame:
            raise HTTPException(
                status_code=404, detail=f"Frame with ID {frame_id} not found.")

        frame_dict = frame.model_dump()
        frame_dict['score'] = score
        frame_dict['selected'] = not selected

        query_key = f"query:{request.query_params.get('query')}:frames"

        if frame_dict['selected']:
            redis_service.add_to_set(query_key, json.dumps(frame_dict))
        else:
            redis_service.remove_from_set(query_key, json.dumps(frame_dict))

        return templates.TemplateResponse("components/frame_card.html", {"request": request, "frame": frame_dict})
    except HTTPException as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error occurred while toggling frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while toggling the frame. Please try again later.")



@router.get("/get_selected_frames", response_class=HTMLResponse)
async def get_selected_frames(request: Request):
    try:
        query_key = f"query:{request.query_params.get('query')}:frames"
        selected_frames = redis_service.get_set_members(query_key)
        selected_frames_dict = [json.loads(frame) for frame in selected_frames]
        return templates.TemplateResponse("components/selected_frames.html", {"request": request, "frames": selected_frames_dict})
    except Exception as e:
        logger.error(
            f"Error occurred while fetching selected frames: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while fetching selected frames. Please try again later.")


@router.post("/submit")
async def submit(selected_frames: list[str] = Form(...)):
    try:
        logger.info(f"Submitting frames: {selected_frames}")
        await save_to_csv(selected_frames)
        return {"message": "Frames submitted successfully"}
    except Exception as e:
        logger.error(f"Error occurred while submitting frames: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while submitting frames. Please try again later.")
