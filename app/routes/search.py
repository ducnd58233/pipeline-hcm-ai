from typing import List
from fastapi import APIRouter, Form, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.models import FrameMetadataModel, ObjectQuery, QueriesStructure, SearchRequest, Searcher, TagQuery, TextQuery
from app.services.fusion.simple_fusion import SimpleFusion
from app.services.reranker.simple_reranker import SimpleReranker
from app.services.search_service import SearchService
from app.services.search_service_v2 import SearchServiceV2
from app.services.searcher.object_detection_searcher import ObjectDetectionSearcher
from app.services.searcher.tag_searcher import TagSearcher
from app.services.searcher.text_searcher import TextSearcher
from app.services.searcher.text_searcher_v2 import TextSearcherV2
from app.log import logger
from app.utils.embedder.open_clip_embedder import OpenClipEmbedder
from app.utils.indexer import FaissIndexer
from app.utils.query_vectorizer.object_detection_vectorizer import ObjectQueryVectorizer
from app.utils.query_vectorizer.tag_vectorizer import TagQueryVectorizer
from app.utils.query_vectorizer.text_vectorizer import TextQueryVectorizer
from app.utils.search_processor import TextProcessor
from app.utils.data_manager.grid_manager import grid_manager
from app.utils.data_manager.tag_manager import tags_list
from app.utils.data_manager.frame_data_manager import frame_data_manager
from config import Config

logger = logger.getChild(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

indexer = FaissIndexer(index_path=Config.FAISS_BIN_PATH)
feature_shape = (indexer.index.d,)

text_embedder = OpenClipEmbedder(
    model_name=Config.CLIP_MODEL_NAME, feature_shape=feature_shape)
text_processor = TextProcessor()

text_query_vectorizer = TextQueryVectorizer(
    text_embedder, text_processor, indexer)
tag_query_vectorizer = TagQueryVectorizer(text_processor, tags_list)
object_detection_vectorizer = ObjectQueryVectorizer()

text_searcher = TextSearcher(text_query_vectorizer)
text_searcher_v2 = TextSearcherV2(text_query_vectorizer)
tag_searcher = TagSearcher(tag_query_vectorizer)
object_detection_searcher = ObjectDetectionSearcher(
    object_detection_vectorizer)

fusion = SimpleFusion()
reranker = SimpleReranker()

search_service = SearchService(
    text_searcher, object_detection_searcher, tag_searcher, fusion, reranker)
search_service_v2 = SearchServiceV2(
    text_searcher_v2, object_detection_searcher, tag_searcher, fusion, reranker)

weights = {
    'text': 50,
    'object': 50,
    'tag': 50,
}
weight_labels = {
    'text': 'Text Weight',
    'object': 'Object Weight',
    'tag': 'Tag Weight',
}

current_results: List[FrameMetadataModel] = []


@router.get("/weight_adjusters", response_class=HTMLResponse)
async def get_weight_adjusters(request: Request):
    logger.info('Rendering weight adjusters')
    return templates.TemplateResponse("components/weight_adjusters.html", {
        "request": request,
        "text_weight": weights['text'],
        "object_weight": weights['object'],
        "tag_weight": weights['tag'],
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
        label = weight_labels[wid]
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


@router.post("/add_tag", response_class=HTMLResponse)
async def add_tag(request: Request, selected_tags: List[str] = Form(...)):
    return templates.TemplateResponse("components/selected_tags.html", {"request": request, "selected_tags": selected_tags})


@router.post("/remove_tag", response_class=HTMLResponse)
async def remove_tag(request: Request, tag: str = Form(...), selected_tags: List[str] = Form(...)):
    updated_tags = [t for t in selected_tags if t != tag]
    return templates.TemplateResponse("components/selected_tags.html", {"request": request, "selected_tags": updated_tags})


@router.get("/tag_search", response_class=HTMLResponse)
async def tag_search(
    request: Request,
    search: str = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100)
):
    filtered_tags = [tag for tag in tags_list if search.lower()
                     in tag.lower()] if search else tags_list
    start = (page - 1) * per_page
    end = start + per_page
    paginated_tags = filtered_tags[start:end]

    return templates.TemplateResponse("components/tag_search.html", {
        "request": request,
        "tags": paginated_tags,
        "search": search,
        "page": page,
        "per_page": per_page,
        "total": len(filtered_tags),
        "has_more": end < len(filtered_tags)
    })


@router.post("/search", response_class=HTMLResponse)
@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    text_query: str = Query(""),
    selected_tags: List[str] = Query([]),
    use_tag_inference: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=500)
):
    global current_results
    try:
        if page == 1:
            current_results = []

        translated_query = ""
        if text_query:
            translated_query = await text_processor.translate_to_english(text_query)
            logger.info(
                f"Translated query: '{text_query}' to '{translated_query}'")

        object_query = ObjectQuery()
        object_query.objects = grid_manager.get_state()

        tag_query = TagQuery(query="", entities=selected_tags)
        if use_tag_inference and translated_query:
            tag_query.query = translated_query

        queries = QueriesStructure(
            text_searcher=Searcher(query=TextQuery(
                query=translated_query), weight=weights['text']/100) if translated_query else None,
            object_detection_searcher=Searcher(
                query=object_query, weight=weights['object']/100) if object_query.objects else None,
            tag_searcher=Searcher(query=tag_query, weight=weights['tag']/100) if (
                selected_tags or (use_tag_inference and translated_query)) else None,
        )

        search_request = SearchRequest(
            queries=queries,
            page=page,
            per_page=per_page
        )

        logger.info(
            f"Searching with queries: {search_request.queries}, page: {page}")
        results = await search_service.search(search_request.queries, use_tag_inference, page=page, per_page=per_page)
        # results = await search_service_v2.search(search_request.queries, use_tag_inference, page=page, per_page=per_page)
        logger.info(f"Found: {len(results.frames)} results")
        current_results.extend(results.frames)

        context = {
            "request": request,
            "results": results.frames if results else [],
            "text_query": text_query,
            "selected_tags": selected_tags,
            "use_tag_inference": use_tag_inference,
            "object_query": grid_manager.get_state(),
            "text_weight": weights['text'] / 100,
            "object_weight": weights['object'] / 100,
            "tag_weight": weights['tag'] / 100,
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


@router.post("/auto_select_frames", response_class=HTMLResponse)
async def auto_select_frames(request: Request, max_items: int = Form(100)):
    global current_results
    try:
        selected_frames = frame_data_manager.get_selected_frames()
        current_count = len(selected_frames)

        frames_to_select = max(0, min(max_items - current_count, 100))

        for frame in current_results:
            if frames_to_select == 0:
                break
            if not frame.selected:
                frame_data_manager.toggle_frame_selection(
                    frame.id, score=frame.final_score)
                frames_to_select -= 1

        updated_selected_frames = frame_data_manager.get_selected_frames()

        return templates.TemplateResponse("components/selected_frames.html", {
            "request": request,
            "frames": updated_selected_frames
        })
    except Exception as e:
        logger.error(f"Error in auto_select_frames: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while auto-selecting frames")
