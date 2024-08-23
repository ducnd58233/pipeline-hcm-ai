from fastapi import APIRouter, Request, Form, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from app.services.csv_service import save_single_frame_to_csv
from app.services.faiss_service import faiss_service
from app.services.frame_service import frame_service
from app.error import FrameNotFoundError
from app.log import logging_config, set_timezone
import logging

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

frame_card_component = "components/frame_card.html"
selected_frame_component = "components/selected_frames.html"
refresh_all_component = "components/refresh_all.html"
confirm_submit_modal = "modals/confirm_submit.html"

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


set_timezone('Asia/Ho_Chi_Minh')
connections = set()


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


@router.post("/toggle_frame")
async def toggle_frame(request: Request, frame_id: str = Form(...), score: float = Form(0.0), user_id: str = Depends(get_user_id)):
    try:
        logger.info(
            f"Received toggle request for frame_id: {frame_id}, score: {score}")

        _, frame_data = await frame_service.toggle_frame_selection(user_id, frame_id, score)

        response = templates.TemplateResponse(
            frame_card_component, {"request": request, "frame": frame_data})
        response.headers["HX-Trigger"] = "frame-toggled"
        return response

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
async def get_selected_frames(request: Request, user_id: str = Depends(get_user_id)):
    frames = await frame_service.get_selected_frames_list(user_id)
    return templates.TemplateResponse(selected_frame_component, {"request": request, "frames": frames})


@router.get("/get_frame_card/{frame_id}", response_class=HTMLResponse)
async def get_frame_card(frame_id: int, request: Request, user_id: str = Depends(get_user_id)):
    frame = frame_service.get_frame_data(user_id, frame_id)
    return templates.TemplateResponse(frame_card_component, {"request": request, "frame": frame})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Depends(get_user_id)):
    await websocket.accept()
    connections.add(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"WS received data: {data}")
            if data.get("action") == "frame_toggled":
                frame_id = data.get("frame_id")
                frame_data = await frame_service.get_frame_data(frame_id)
                html = templates.render(
                    frame_card_component, {"frame": frame_data})
                await websocket.send_json({
                    "action": "update_frame",
                    "frame_id": frame_id,
                    "html": html
                })
    except WebSocketDisconnect:
        connections.remove(websocket)


@router.post("/submit_single_frame", response_class=HTMLResponse)
async def submit_single_frame(request: Request, frame_id: str = Form(...)):
    return templates.TemplateResponse("modals/confirm_submit.html", {"request": request, "frame_id": frame_id})


@router.post("/confirm_submit_single", response_class=HTMLResponse)
async def confirm_submit_single(request: Request, frame_id: str = Form(...), user_id: str = Depends(get_user_id)):
    try:
        frame = await frame_service.get_frame_data(frame_id)
        await save_single_frame_to_csv(frame)
        await frame_service.clear_all_selected_frames(user_id)

        response = templates.TemplateResponse(
            "index.html", {"request": request})
        response.headers["HX-Trigger"] = "refreshAll"
        return response
    except Exception as e:
        logger.error(f"Error in confirm_submit_single: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your submission")


@router.get("/refresh_components", response_class=HTMLResponse)
async def refresh_components(request: Request, user_id: str = Depends(get_user_id)):
    selected_frames = await frame_service.get_selected_frames_list(user_id)
    return templates.TemplateResponse("components/refresh_all.html", {
        "request": request,
        "selected_frames": selected_frames
    })

@router.get("/close_modal", response_class=HTMLResponse)
async def close_modal():
    return ""
