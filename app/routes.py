from typing import Dict, Tuple
from fastapi import APIRouter, Query, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.services.search_service import SearchService
from app.services.searcher.object_detection_searcher import ObjectDetectionSearcher
from app.services.searcher.text_searcher import TextSearcher
from app.utils.frame_data_manager import frame_data_manager
from app.utils.grid_manager import grid_manager
from app.error import FrameNotFoundError
from app.log import logging_config, set_timezone
import logging
from app.utils.indexer import FaissIndexer
from app.utils.relevance_calculator import RelevanceCalculator
from app.utils.reranker import Reranker
from app.utils.search_processor import TextProcessor
from app.utils.vectorizer import OpenClipVectorizer
from app.services.csv_service import save_single_frame_to_csv

from config import Config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
feature_shape = (indexer.index.d,)


frame_card_component = "components/frame_card.html"
search_results_component = "components/search_results.html"
selected_frame_component = "components/selected_frames.html"
object_query_input_component = "components/object_query_input.html"
refresh_all_component = "components/refresh_all.html"
drag_drop_panel_component = "components/drag_drop_panel.html"
grid_cell_component = "components/grid_cell.html"

confirm_submit_modal = "modals/confirm_submit.html"


# Global state
global_panel_state = "enabled"
global_panel_logic = "and"
global_max_objects = ""
grid_state: Dict[Tuple[int, int], str] = {(0, 0): 'cat'}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

set_timezone('Asia/Ho_Chi_Minh')


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/start_drag", response_class=HTMLResponse)
async def start_drag(request: Request, category: str = Form(...)):
    logger.info(f"Starting drag for category: {category}")
    return f'<input type="hidden" name="category" value="{category}">'


@router.post("/add_object_to_grid", response_class=HTMLResponse)
async def add_object_to_grid(request: Request, row: int = Form(...), col: int = Form(...), category: str = Form(...)):
    grid_manager.add_object(row, col, category)

    return templates.TemplateResponse("components/grid_cell.html", {
        "request": request,
        "row": row,
        "col": col,
        "grid_state": grid_manager.get_state()
    })


@router.post("/remove_object_from_grid", response_class=HTMLResponse)
async def remove_object_from_grid(request: Request, row: int = Form(...), col: int = Form(...)):
    grid_manager.remove_object(row, col)

    return templates.TemplateResponse("components/grid_cell.html", {
        "request": request,
        "row": row,
        "col": col,
        "grid_state": grid_manager.get_state()
    })


@router.post("/update_panel_logic", response_class=HTMLResponse)
async def update_panel_logic(request: Request, panel_logic: str = Form(...)):
    global global_panel_logic
    global_panel_logic = panel_logic
    logger.info(f"Updated panel logic to: {panel_logic}")
    return await get_drag_drop_panel(request)


@router.post("/update_max_objects", response_class=HTMLResponse)
async def update_max_objects(request: Request, max_objects: str = Form(...)):
    global global_max_objects
    global_max_objects = max_objects
    logger.info(f"Updated max objects to: {max_objects}")
    return await get_drag_drop_panel(request)


@router.get("/drag_drop_panel", response_class=HTMLResponse)
async def get_drag_drop_panel(request: Request):
    categories = ["airplane", "bicycle",
                  "bird", "boat", "cat", "dog", "person"]
    logger.info("Rendering drag drop panel")
    logger.debug(f"Current grid state: {grid_state}")
    return templates.TemplateResponse("components/drag_drop_panel.html", {
        "request": request,
        "categories": categories,
        "panel_state": global_panel_state,
        "panel_logic": global_panel_logic,
        "grid_state": grid_state,
        "max_objects": global_max_objects
    })


@router.post("/clear_objects", response_class=HTMLResponse)
async def clear_objects(request: Request):
    global grid_state
    grid_state.clear()
    logger.info("Cleared all objects from grid")
    return await get_drag_drop_panel(request)


@router.post("/reset_panel", response_class=HTMLResponse)
async def reset_panel(request: Request):
    global global_panel_state, global_panel_logic, grid_state, global_max_objects
    global_panel_state = "enabled"
    global_panel_logic = "and"
    grid_state.clear()
    global_max_objects = ""
    logger.info("Reset panel to default state")
    return await get_drag_drop_panel(request)



@router.get("/add_object_query", response_class=HTMLResponse)
async def add_object_query(request: Request):
    return templates.TemplateResponse(object_query_input_component, {"request": request})


@router.get("/remove_object_query", response_class=HTMLResponse)
async def remove_object_query():
    return ""


@router.post("/toggle_frame", response_class=HTMLResponse)
async def toggle_frame(request: Request, frame_id: str = Form(...), score: float = Form(0.0)):
    try:
        logger.info(
            f"Received toggle request for frame_id: {frame_id}, score: {score}")
        frame_data_manager.toggle_frame_selection(
            frame_id, score)
        frame_data = frame_data_manager.get_frame_by_id(frame_id)

        if frame_data is None:
            raise FrameNotFoundError(f"Frame not found: {frame_id}")

        response = templates.TemplateResponse(
            frame_card_component, {"request": request, "frame": frame_data})
        response.headers["HX-Trigger"] = "frame-toggled"
        return response
    except FrameNotFoundError as e:
        logger.error(f"Frame not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in toggle_frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")


@router.get("/get_selected_frames", response_class=HTMLResponse)
async def get_selected_frames(request: Request):
    frames = frame_data_manager.get_selected_frames()
    return templates.TemplateResponse(selected_frame_component, {"request": request, "frames": frames})


@router.get("/get_frame_card/{frame_id}", response_class=HTMLResponse)
async def get_frame_card(frame_id: str, request: Request):
    frame = frame_data_manager.get_frame_by_id(frame_id)
    if frame is None:
        raise HTTPException(
            status_code=404, detail=f"Frame not found: {frame_id}")
    return templates.TemplateResponse(frame_card_component, {"request": request, "frame": frame})


@router.post("/submit_single_frame", response_class=HTMLResponse)
async def submit_single_frame(request: Request, frame_id: str = Form(...)):
    return templates.TemplateResponse(confirm_submit_modal, {"request": request, "frame_id": frame_id})


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
    return templates.TemplateResponse(refresh_all_component, {
        "request": request,
        "selected_frames": selected_frames
    })


@router.get("/close_modal", response_class=HTMLResponse)
async def close_modal():
    return ""
