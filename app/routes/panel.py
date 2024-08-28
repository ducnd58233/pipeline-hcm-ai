from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.utils.grid_manager import grid_manager
from app.log import logger
from app.models import Category

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.post("/update_panel_logic", response_class=HTMLResponse)
async def update_panel_logic(request: Request, panel_logic: str = Query(...)):
    try:
        grid_manager.set_panel_logic(panel_logic)
        logger.info(f"Updated panel logic to: {panel_logic}")
    except ValueError as e:
        logger.error(f"Invalid panel logic: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    return await get_drag_drop_panel(request)


@router.post("/update_max_objects", response_class=HTMLResponse)
async def update_max_objects(request: Request, max_objects: str = Query(...)):
    grid_manager.set_max_objects(max_objects)
    logger.info(f"Updated max objects to: {max_objects}")
    return await get_drag_drop_panel(request)


@router.get("/drag_drop_panel", response_class=HTMLResponse)
async def get_drag_drop_panel(request: Request):
    categories = [category.value for category in Category]
    grid_state_values = {k: v.value for k,
                         v in grid_manager.get_state().items()}
    return templates.TemplateResponse("components/drag_drop_panel.html", {
        "request": request,
        "categories": categories,
        "panel_state": "enabled",  # You might want to make this dynamic
        "panel_logic": grid_manager.get_panel_logic(),
        "grid_state": grid_state_values,
        "max_objects": grid_manager.get_max_objects()
    })
