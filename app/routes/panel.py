from math import ceil
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.utils.icon_mapper import icon_map
from app.utils.data_manager.grid_manager import grid_manager
from app.log import logger
from app.models import Category

logger = logger.getChild(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

ITEMS_PER_PAGE = 10


@router.get("/search_icons", response_class=HTMLResponse)
async def search_icons(request: Request, query: str = Query(None)):
    if query:
        matching_categories = [
            category for category in Category
            if query.lower() in category.name.lower()
        ]
    else:
        matching_categories = []

    return templates.TemplateResponse("components/icon_suggestions.html", {
        "request": request,
        "categories": matching_categories,
        "icon_map": icon_map
    })


@router.get("/icon_search_bar", response_class=HTMLResponse)
async def get_icon_search_bar(request: Request):
    return templates.TemplateResponse("components/icon_search_bar.html", {
        "request": request
    })

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
    categories = list(Category)
    grid_state_values = {k: v.value for k,
                         v in grid_manager.get_state().items()}
    return templates.TemplateResponse("components/drag_drop_panel.html", {
        "request": request,
        "categories": categories,
        "panel_state": "enabled",
        "icon_map": icon_map,
        "panel_logic": grid_manager.get_panel_logic(),
        "grid_state": grid_state_values,
        "max_objects": grid_manager.get_max_objects()
    })


@router.get("/paginated_icon_grid", response_class=HTMLResponse)
async def get_paginated_icon_grid(request: Request, page: int = Query(1, ge=1)):
    categories = list(Category)
    total_categories = len(categories)
    total_pages = ceil(total_categories / ITEMS_PER_PAGE)

    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_categories = categories[start_index:end_index]

    return templates.TemplateResponse("components/paginated_icon_grid.html", {
        "request": request,
        "categories": paginated_categories,
        "icon_map": icon_map,
        "page": page,
        "total_pages": total_pages,
        "selected_category": None
    })
