from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.utils.grid_manager import grid_manager
from app.log import logger

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

global_panel_state = "enabled"
global_panel_logic = "and"
global_max_objects = ""


@router.get("/drag_drop_panel", response_class=HTMLResponse)
async def get_drag_drop_panel(request: Request):
    categories = ["airplane", "bicycle",
                  "bird", "boat", "cat", "dog", "person"]
    logger.info("Rendering drag drop panel")
    logger.debug(f"Current grid state: {grid_manager.get_state()}")
    return templates.TemplateResponse("components/drag_drop_panel.html", {
        "request": request,
        "categories": categories,
        "panel_state": global_panel_state,
        "panel_logic": global_panel_logic,
        "grid_state": grid_manager.get_state(),
        "max_objects": global_max_objects
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


@router.post("/clear_objects", response_class=HTMLResponse)
async def clear_objects(request: Request):
    grid_manager.clear()
    logger.info("Cleared all objects from grid")
    return await get_drag_drop_panel(request)


@router.post("/reset_panel", response_class=HTMLResponse)
async def reset_panel(request: Request):
    global global_panel_state, global_panel_logic, global_max_objects
    global_panel_state = "enabled"
    global_panel_logic = "and"
    global_max_objects = ""
    grid_manager.clear()
    logger.info("Reset panel to default state")
    return await get_drag_drop_panel(request)
