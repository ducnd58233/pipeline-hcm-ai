from fastapi import APIRouter, Request, Form, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from app.services.faiss_service import faiss_service
from app.services.frame_service import frame_service
from app.error import FrameNotFoundError
from app.log import logging_config, set_timezone
import logging

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


set_timezone('Asia/Ho_Chi_Minh')


def get_user_id():
    return "default_user"


@router.post("/set_timezone")
async def set_timezone_route(timezone: str):
    try:
        set_timezone(timezone)
        frame_service.set_timezone(timezone)
        return JSONResponse(content={"message": f"Timezone set to {timezone}"}, status_code=200)
    except ValueError as e:
        logger.error(f"Invalid timezone: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in set_timezone: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = "", page: int = 1, per_page: int = 20):
    try:
        if query.strip():
            logger.info(f"Searching for query: {query}, page: {page}")
            results = await faiss_service.search(query, page=page, per_page=per_page, extra_k=50)
            total_results = len(results)
            total_pages = (total_results + per_page - 1) // per_page
        else:
            results = []
            total_results = 0
            total_pages = 0

        return templates.TemplateResponse("components/search_results.html", {
            "request": request,
            "results": results,
            "page": page,
            "total_pages": total_pages,
            "query": query,
            "per_page": per_page,
            "total_results": total_results
        })
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while searching. Please try again later.")


@router.post("/search", response_class=HTMLResponse)
async def search_post(request: Request, query: str = Form(...), page: int = Form(1), per_page: int = Form(20)):
    return await search(request, query, page, per_page)


@router.post("/toggle_frame", response_class=HTMLResponse)
async def toggle_frame(request: Request, frame_id: str = Form(...), score: float = Form(0.0), user_id: str = Depends(get_user_id)):
    try:
        logger.info(
            f"Received toggle request for frame_id: {frame_id}, score: {score}")
        
        await frame_service.toggle_frame_selection(user_id, frame_id, score)

        selected_frames = frame_service.get_selected_frames_list(user_id)

        return templates.TemplateResponse("components/selected_frames.html", {
            "request": request,
            "frames": selected_frames
        })
    except FrameNotFoundError as e:
        logger.error(f"Frame not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in toggle_frame: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")


@router.get("/get_selected_frames", response_class=HTMLResponse)
def get_selected_frames(request: Request, user_id: str = Depends(get_user_id)):
    frames = frame_service.get_selected_frames_list(user_id)
    return templates.TemplateResponse("components/selected_frames.html", {"request": request, "frames": frames})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Depends(get_user_id)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data['action'] == 'select_frame':
                frame_id = data['frame_id']
                await frame_service.toggle_frame_selection(user_id, frame_id)
                await websocket.send_json({"action": "update_selected_frames"})
    except WebSocketDisconnect:
        pass


@router.post("/submit_frame", response_class=HTMLResponse)
async def submit_frame(request: Request, user_id: str = Depends(get_user_id)):
    return templates.TemplateResponse("modals/confirm_submit.html", {"request": request})


@router.post("/confirm_submit", response_class=HTMLResponse)
async def confirm_submit(request: Request, user_id: str = Depends(get_user_id)):
    await frame_service.clear_all_selected_frames(user_id)
    return templates.TemplateResponse("components/refresh_all.html", {"request": request})
