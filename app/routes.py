from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.services.search_service import SearchService
from app.utils.frame_data_manager import frame_data_manager
from app.error import FrameNotFoundError
from app.log import logging_config, set_timezone
import logging
from app.utils.indexer import FaissIndexer
from app.utils.relevance_calculator import EnhancedRelevanceCalculator
from app.utils.reranker import EnhancedReranker
from app.utils.text_processor import TextProcessor
from app.utils.vectorizer import ClipVectorizer
from app.services.csv_service import save_single_frame_to_csv

from config import Config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

vectorizer = ClipVectorizer()
indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
text_processor = TextProcessor()
relevance_calculator = EnhancedRelevanceCalculator(text_processor)
reranker = EnhancedReranker(relevance_calculator)

search_service = SearchService(vectorizer, indexer, reranker)

frame_card_component = "components/frame_card.html"
search_results_component = "components/search_results.html"
selected_frame_component = "components/selected_frames.html"
refresh_all_component = "components/refresh_all.html"
confirm_submit_modal = "modals/confirm_submit.html"

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

set_timezone('Asia/Ho_Chi_Minh')


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = "", page: int = 1, per_page: int = 20):
    try:
        results = []
        if query.strip():
            logger.info(f"Searching for query: {query}, page: {page}")
            offset = (page - 1) * per_page
            results = await search_service.search(query, offset=offset, limit=per_page)
            logger.info(f"Found: {len(results)} results")

        context = {
            "request": request,
            "results": results,
            "query": query,
            "page": page,
            "per_page": per_page
        }
        logger.debug(f"Context for template: {context}")

        return templates.TemplateResponse(search_results_component, context)
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while searching. Please try again later.")


@router.post("/search", response_class=HTMLResponse)
async def search_post(request: Request, query: str = Form(...), page: int = Form(1), per_page: int = Form(20)):
    return await search(request, query, page, per_page)


@router.post("/toggle_frame", response_class=HTMLResponse)
async def toggle_frame(request: Request, frame_id: str = Form(...), final_score: float = Form(0.0)):
    try:
        logger.info(
            f"Received toggle request for frame_id: {frame_id}, score: {final_score}")
        _ = frame_data_manager.toggle_frame_selection(
            frame_id, final_score)
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
