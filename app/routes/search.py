from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.services.search_service import SearchService
from app.services.searcher.object_detection_searcher import ObjectDetectionSearcher
from app.services.searcher.text_searcher import TextSearcher
from app.log import logger
from app.utils.indexer import FaissIndexer
from app.utils.search_processor import TextProcessor
from app.utils.vectorizer import OpenClipVectorizer
from app.utils.grid_manager import grid_manager
from config import Config

logger = logger.getChild(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
feature_shape = (indexer.index.d,)

text_processor = TextProcessor()

vectorizer = OpenClipVectorizer(feature_shape=feature_shape)
text_searcher = TextSearcher(vectorizer, indexer, text_processor)
object_detection_searcher = ObjectDetectionSearcher(text_processor)

searchers = {
    'text': text_searcher,
    'object': object_detection_searcher
}

weights = {
    'text': 0.5,
    'object': 0.5
}

search_service = SearchService(searchers, weights)


@router.post("/search", response_class=HTMLResponse)
@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = "",
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=1000)
):
    try:
        logger.info(
            f"Searching for query: {query}, object_query: {grid_manager.get_state()}, page: {page}")
        results = await search_service.search(query, grid_manager.get_state(), page=page, per_page=per_page)
        logger.info(f"Found: {len(results.frames)} results")

        context = {
            "request": request,
            "results": results.frames if results else [],
            "query": query,
            "page": page,
            "per_page": per_page,
            "total": results.total if results else 0,
            "has_more": results.has_more if results else False
        }
        logger.debug(f"Context for template: {context}")

        return templates.TemplateResponse("components/search_results.html", context)
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while searching. Please try again later.")
