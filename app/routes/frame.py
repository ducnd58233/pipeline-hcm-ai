from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Optional
from app.utils.data_manager.frame_data_manager import frame_data_manager
from app.services.csv_service import (
    save_single_frame_to_csv, get_existing_csv_files, create_new_csv_file,
    get_file_contents, add_frame_to_file, remove_frame_from_file
)
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

        frame = frame_data_manager.get_frame_by_key(frame_id)
        selected_frames = frame_data_manager.get_selected_frames()

        frame_card_html = templates.TemplateResponse(
            "components/frame_card.html",
            {"request": request, "frame": frame}
        ).body.decode()

        selected_frames_html = templates.TemplateResponse(
            "components/selected_frames.html",
            {"request": request, "frames": selected_frames}
        ).body.decode()

        return f"""
        <div id="frame-{frame_id}" hx-swap-oob="true">{frame_card_html}</div>
        <div id="frame-{frame_id}-search-result" hx-swap-oob="true">{frame_card_html}</div>
        <div id="selected-frames-container" hx-swap-oob="true">{selected_frames_html}</div>
        """
    except FrameNotFoundError as e:
        logger.error(f"Frame not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in toggle_frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")


@router.get("/get_frame_card/{frame_id}", response_class=HTMLResponse)
async def get_frame_card(request: Request, frame_id: str):
    frame = frame_data_manager.get_frame_by_key(frame_id)
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
    existing_files = get_existing_csv_files()
    return templates.TemplateResponse("modals/confirm_submit.html", {
        "request": request,
        "frame_id": frame_id,
        "existing_files": existing_files
    })


@router.post("/submit_all_frames", response_class=HTMLResponse)
async def submit_all_frames(request: Request):
    selected_frames = frame_data_manager.get_selected_frames()
    existing_files = get_existing_csv_files()
    return templates.TemplateResponse("modals/confirm_submit_all.html", {
        "request": request,
        "frame_count": len(selected_frames),
        "existing_files": existing_files
    })


@router.get("/get_file_contents", response_class=HTMLResponse)
async def get_file_contents_route(request: Request, file_name: str):
    contents = get_file_contents(file_name)
    return templates.TemplateResponse("components/file_contents.html", {
        "request": request,
        "contents": contents,
        "file_name": file_name
    })


@router.post("/create_file", response_class=HTMLResponse)
async def create_file(request: Request, new_file_name: str = Form(...)):
    file_name = create_new_csv_file(new_file_name)
    existing_files = get_existing_csv_files()
    return templates.TemplateResponse("components/file_select.html", {
        "request": request,
        "existing_files": existing_files,
        "selected_file": file_name
    })


@router.post("/add_frame_to_file", response_class=HTMLResponse)
async def add_frame_to_file_route(request: Request, file_name: str = Form(...), frame_id: str = Form(...)):
    frame = frame_data_manager.get_frame_by_key(frame_id)
    if frame is None:
        raise HTTPException(
            status_code=404, detail=f"Frame not found: {frame_id}")
    add_frame_to_file(frame, file_name)
    contents = get_file_contents(file_name)
    return templates.TemplateResponse("components/file_contents.html", {
        "request": request,
        "contents": contents,
        "file_name": file_name
    })


@router.post("/remove_frame_from_file", response_class=HTMLResponse)
async def remove_frame_from_file_route(request: Request, file_name: str = Form(...), frame_id: str = Form(...)):
    remove_frame_from_file(frame_id, file_name)
    contents = get_file_contents(file_name)
    return templates.TemplateResponse("components/file_contents.html", {
        "request": request,
        "contents": contents,
        "file_name": file_name
    })


@router.post("/confirm_submit_single", response_class=HTMLResponse)
async def confirm_submit_single(
    request: Request,
    frame_id: str = Form(...),
    file_name: str = Form(...),
    new_file_name: Optional[str] = Form(None)
):
    try:
        frame = frame_data_manager.get_frame_by_key(frame_id)
        if frame is None:
            raise FrameNotFoundError(f"Frame not found: {frame_id}")

        if new_file_name:
            file_name = create_new_csv_file(new_file_name)

        await save_single_frame_to_csv(frame, file_name)
        frame_data_manager.remove_frame_selection(frame_id)

        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error in confirm_submit_single: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your submission")


@router.post("/confirm_submit_all", response_class=HTMLResponse)
async def confirm_submit_all(
    request: Request,
    file_name: Optional[str] = Form(None),
    new_file_name: Optional[str] = Form(None)
):
    try:
        selected_frames = frame_data_manager.get_selected_frames()

        if new_file_name:
            file_name = create_new_csv_file(new_file_name)
        elif not file_name:
            raise HTTPException(
                status_code=400, detail="Either file_name or new_file_name must be provided")

        for frame in selected_frames:
            await save_single_frame_to_csv(frame, file_name)
            frame_data_manager.remove_frame_selection(frame.id)

        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error in confirm_submit_all: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your submission")


@router.get("/get_file_contents", response_class=HTMLResponse)
async def get_file_contents_route(request: Request, file_name: str):
    try:
        contents = get_file_contents(file_name)
        return templates.TemplateResponse("components/file_contents.html", {
            "request": request,
            "contents": contents,
            "file_name": file_name
        })
    except Exception as e:
        logger.error(f"Error in get_file_contents: {str(e)}")
        return HTMLResponse(content=f"Error loading file contents: {str(e)}", status_code=500)


@router.post("/create_file", response_class=HTMLResponse)
async def create_file(request: Request, new_file_name: str = Form(...)):
    file_name = create_new_csv_file(new_file_name)
    existing_files = get_existing_csv_files()
    return templates.TemplateResponse("components/file_select.html", {
        "request": request,
        "existing_files": existing_files,
        "selected_file": file_name
    })


@router.get("/close_modal", response_class=HTMLResponse)
async def close_modal():
    return ""
