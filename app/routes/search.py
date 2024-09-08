from fastapi import APIRouter, Depends, Form, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.models import ObjectQuery, QueriesStructure, SearchRequest, Searcher, TextQuery
from app.services.fusion.simple_fusion import SimpleFusion
from app.services.reranker.sbert_reranker import SentenceBertReranker
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
from app.utils.data_manager.grid_manager import grid_manager
from config import Config

logger = logger.getChild(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
feature_shape = (indexer.index.d,)

text_embedder = OpenClipEmbedder(model_name=Config.CLIP_MODEL_NAME, feature_shape=feature_shape)
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

search_service = SearchService(
    text_searcher, object_detection_searcher, fusion, reranker)

weights = {
    'text': 50,
    'object': 50,
}


@router.get("/weight_adjusters", response_class=HTMLResponse)
async def get_weight_adjusters(request: Request):
    logger.info('Rendering weight adjusters')
    return templates.TemplateResponse("components/weight_adjusters.html", {
        "request": request,
        "text_weight": weights['text'],
        "object_weight": weights['object'],
    })


@router.post("/update_weight", response_class=HTMLResponse)
async def update_weight(request: Request):
    form_data = await request.form()
    logger.info(f"Received form data: {form_data}")

    try:
        wid = form_data.get("id")
        value = int(form_data.get(
            f"weight-{wid}", form_data.get("weight_input", "0")))

        value = max(0, min(100, value))
        weights[wid] = value
        label = "Text Weight" if wid == "text" else "Object Weight"
        logger.info(f'Updating {wid} weight to {value}')

        response = templates.TemplateResponse("components/weight_adjuster.html", {
            "request": request,
            "id": wid,
            "value": value,
            "label": label
        })

        response.headers["HX-Trigger"] = "weight-changed"
        return response
    except Exception as e:
        logger.error(f"Error in update_weight: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid input data")


@router.post("/search", response_class=HTMLResponse)
@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    text_query: str = Query(""),
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=500)
):
    try:
        queries = QueriesStructure(
            text_searcher=Searcher(query=TextQuery(
                query=text_query), weight=weights['text']/100) if text_query else None,
            object_detection_searcher=None
        )
        
        object_query = ObjectQuery()
        object_query.objects = grid_manager.get_state()

        if object_query.objects:
            queries.object_detection_searcher = Searcher(
                query=object_query, weight=weights['object'] / 100)

        search_request = SearchRequest(
            queries=queries,
            page=page,
            per_page=per_page
        )

        logger.info(
            f"Searching with queries: {search_request.queries}, page: {page}")
        results = await search_service.search(search_request.queries, page=page, per_page=per_page)
        logger.info(f"Found: {len(results.frames)} results")

        context = {
            "request": request,
            "results": results.frames if results else [],
            "text_query": queries.text_searcher.query.query if queries.text_searcher else "",
            "object_query": grid_manager.get_state(),
            "text_weight": weights['text'] / 100,
            "object_weight": weights['object'] / 100,
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
