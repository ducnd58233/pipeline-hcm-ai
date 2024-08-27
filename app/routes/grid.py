from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.utils.grid_manager import grid_manager
from app.log import logger

logger = logger.getChild(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
current_selected_category = ''


@router.post("/select_category", response_class=HTMLResponse)
async def select_category(request: Request, category: str = Query(...)):
    global current_selected_category
    
    try:
        logger.info(f"Selected category: {category}")
        current_selected_category = category
        return f'<input type="hidden" name="category" value="{category}">{category} selected'
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    


@router.post("/add_object_to_grid", response_class=HTMLResponse)
async def add_object_to_grid(
    request: Request,
    row: int = Query(...),
    col: str = Query(...),
):
    global current_selected_category
    logger.info(
        f"Adding object to grid: row={row}, col={col}, category={current_selected_category}")

    try:
        if not current_selected_category:
            logger.error('No select category found')
            raise HTTPException(status_code=404, detail='Need to select category first')
        grid_manager.add_object(row, col, current_selected_category)
        current_selected_category = ''

        return templates.TemplateResponse("components/grid_cell.html", {
            "request": request,
            "row": row,
            "col": col,
            "grid_state": grid_manager.get_state()
        })
    except Exception as e:
        logger.error(f"Error in add_object_to_grid: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/remove_object_from_grid", response_class=HTMLResponse)
async def remove_object_from_grid(
    request: Request,
    row: int = Query(...),
    col: str = Query(...)
):
    logger.info(f"Removing object from grid: row={row}, col={col}")
    grid_manager.remove_object(row, col)

    return templates.TemplateResponse("components/grid_cell.html", {
        "request": request,
        "row": row,
        "col": col,
        "grid_state": grid_manager.get_state()
    })
