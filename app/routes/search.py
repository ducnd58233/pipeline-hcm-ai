from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.models import ObjectQuery, SearchRequest
from app.services.fusion.simple_fusion import SimpleFusion
from app.services.reranker.simple_reranker import SimpleReranker
from app.services.search_service import SearchService
from app.services.searcher.object_detection_searcher import ObjectDetectionSearcher
from app.services.searcher.text_searcher import TextSearcher
from app.log import logger
from app.utils.embedder.open_clip_embedder import OpenClipEmbedder
from app.utils.indexer import FaissIndexer
from app.utils.query_vectorizer.object_detection_vectorizer import ObjectQueryVectorizer
from app.utils.query_vectorizer.text_vectorizer import TextQueryVectorizer
from app.utils.search_processor import TextProcessor
from app.utils.grid_manager import grid_manager
from config import Config

logger = logger.getChild(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
feature_shape = (indexer.index.d,)

text_embedder = OpenClipEmbedder(feature_shape=feature_shape)
text_processor = TextProcessor()

text_query_vectorizer = TextQueryVectorizer(text_embedder, text_processor)
object_detection_vectorizer = ObjectQueryVectorizer()

text_searcher = TextSearcher(text_query_vectorizer, indexer)
object_detection_searcher = ObjectDetectionSearcher(
    object_detection_vectorizer)

searchers = {
    'text': text_searcher,
    'object': object_detection_searcher
}
fusion = SimpleFusion()
reranker = SimpleReranker()

search_service = SearchService(searchers, fusion, reranker)


async def get_search_params(
    text_query: str = Query(""),
    text_weight: float = Query(0.5, ge=0.0, le=1.0),
    object_weight: float = Query(0.5, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    return {
        "text_query": text_query,
        "text_weight": text_weight,
        "object_weight": object_weight,
        "page": page,
        "per_page": per_page
    }


@router.post("/search", response_class=HTMLResponse)
@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    params: dict = Depends(get_search_params)
):
    try:
        obj_query = ObjectQuery()
        obj_query.objects = grid_manager.get_state()

        search_request = SearchRequest(
            text_query=params["text_query"],
            objects_query=obj_query,
            page=params["page"],
            per_page=params["per_page"]
        )

        queries = {
            'text': search_request.text_query,
            'object': search_request.objects_query,
        }

        weights = {
            'text': params["text_weight"],
            'object': params["object_weight"],
        }

        logger.info(
            f"Searching with queries: {queries}, weights: {weights}, page: {params['page']}")
        results = await search_service.search(queries, weights, page=params["page"], per_page=params["per_page"])
        logger.info(f"Found: {len(results.frames)} results")

        context = {
            "request": request,
            "results": results.frames if results else [],
            "text_query": params["text_query"],
            "object_query": grid_manager.get_state(),
            "text_weight": params["text_weight"],
            "object_weight": params["object_weight"],
            "page": int(params["page"]),  # Ensure page is an integer
            "per_page": params["per_page"],
            "total": results.total if results else 0,
            "has_more": results.has_more if results else False
        }
        logger.debug(f"Context for template: {context}")

        if int(params["page"]) > 1:
            return templates.TemplateResponse("components/frame_cards.html", context)
        else:
            return templates.TemplateResponse("components/search_results.html", context)
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while searching. Please try again later.")
