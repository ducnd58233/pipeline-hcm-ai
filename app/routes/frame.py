from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.utils.frame_data_manager import frame_data_manager
from app.services.csv_service import save_single_frame_to_csv
from app.error import FrameNotFoundError
from app.log import logger

logger = logger.getChild(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.post("/toggle_frame", response_class=HTMLResponse)
async def toggle_frame(request: Request, frame_id: str = Form(...), score: float = Form(0.0)):
    try:
        logger.info(
            f"Received toggle request for frame_id: {frame_id}, score: {score}")
        frame_data_manager.toggle_frame_selection(frame_id, score)
        frame_data = frame_data_manager.get_frame_by_id(frame_id)

        if frame_data is None:
            raise FrameNotFoundError(f"Frame not found: {frame_id}")

        response = templates.TemplateResponse(
            "components/frame_card.html", {"request": request, "frame": frame_data})
        response.headers["HX-Trigger"] = "frame-toggled"
        return response
    except FrameNotFoundError as e:
        logger.error(f"Frame not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in toggle_frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")


@router.get("/get_frame_card/{frame_id}", response_class=HTMLResponse)
async def get_frame_card(request: Request, frame_id: str):
    frame = frame_data_manager.get_frame_by_id(frame_id)
    if frame is None:
        raise HTTPException(
            status_code=404, detail=f"Frame not found: {frame_id}")
    return templates.TemplateResponse("components/frame_card.html", {"request": request, "frame": frame})


@router.get("/get_selected_frames", response_class=HTMLResponse)
async def get_selected_frames(request: Request):
    frames = frame_data_manager.get_selected_frames()
    return templates.TemplateResponse("components/selected_frames.html", {"request": request, "frames": frames})


@router.post("/submit_single_frame", response_class=HTMLResponse)
async def submit_single_frame(request: Request, frame_id: str = Form(...)):
    return templates.TemplateResponse("modals/confirm_submit.html", {"request": request, "frame_id": frame_id})


@router.post("/confirm_submit_single", response_class=HTMLResponse)
async def confirm_submit_single(request: Request, frame_id: str = Form(...)):
    try:
        frame = frame_data_manager.get_frame_by_id(frame_id)
        if frame is None:
            raise FrameNotFoundError(f"Frame not found: {frame_id}")
        await save_single_frame_to_csv(frame)
        frame_data_manager.clear_all()

        response = templates.TemplateResponse(
            "index.html", {"request": request})
        response.headers["HX-Trigger"] = "refreshAll"
        return response
    except Exception as e:
        logger.error(f"Error in confirm_submit_single: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your submission")


@router.get("/refresh_components", response_class=HTMLResponse)
async def refresh_components(request: Request):
    selected_frames = frame_data_manager.get_selected_frames()
    return templates.TemplateResponse("components/refresh_all.html", {
        "request": request,
        "selected_frames": selected_frames
    })


@router.get("/close_modal", response_class=HTMLResponse)
async def close_modal():
    return ""
