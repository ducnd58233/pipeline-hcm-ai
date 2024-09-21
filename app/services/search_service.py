import asyncio
from typing import List, Optional, Dict
from app.models import SearchResult, FrameMetadataModel, QueriesStructure, TextQuery
from app.services.searcher.object_detection_searcher import ObjectDetectionSearcher
from app.services.searcher.tag_searcher import TagSearcher
from app.services.searcher.text_searcher import TextSearcher
from app.services.fusion.abstract_fusion import AbstractFusion
from app.services.reranker.abstract_reranker import AbstractReranker
from app.log import logger

logger = logger.getChild(__name__)


class SearchService:
    def __init__(self,
                 text_searcher: TextSearcher,
                 object_detection_searcher: ObjectDetectionSearcher,
                 tag_searcher: TagSearcher,
                 fusion: AbstractFusion,
                 reranker: AbstractReranker,
                 ):
        self.text_searcher = text_searcher
        self.tag_searcher = tag_searcher
        self.object_detection_searcher = object_detection_searcher
        self.fusion = fusion
        self.reranker = reranker

    async def search(self, queries: QueriesStructure, use_tag_inference: bool, page: int = 1, per_page: int = 20, boost_factors: Optional[Dict[str, float]] = None) -> SearchResult:
        try:
            logger.info(
                f"Performing search for queries: {queries}, use_tag_inference: {use_tag_inference}, page: {page}, per_page: {per_page}")

            if not queries.text_searcher and not queries.tag_searcher and not queries.object_detection_searcher:
                logger.warning("No active queries provided.")
                return SearchResult(frames=[], total=0, page=page, has_more=False)

            search_tasks = []
            searcher_results = {}

            if queries.text_searcher:
                search_tasks.append(self.text_searcher.search(
                    queries.text_searcher.query, page=page, per_page=page*per_page))

            if queries.tag_searcher:
                search_tasks.append(self.tag_searcher.search(
                    queries.tag_searcher.query, page=page, per_page=page*per_page, boost_factors=boost_factors))

            if queries.object_detection_searcher:
                search_tasks.append(self.object_detection_searcher.search(
                    queries.object_detection_searcher.query, page=page, per_page=page*per_page))

            results = await asyncio.gather(*search_tasks)

            if queries.text_searcher:
                searcher_results['text'] = results.pop(0)

            if queries.tag_searcher:
                searcher_results['tag'] = results.pop(0)

            if queries.object_detection_searcher:
                searcher_results['object'] = results.pop(0)

            logger.debug(f'Found: {searcher_results}')

            merged_results = self.fusion.merge_results(
                searcher_results, queries)

            text_query = queries.text_searcher.query if queries.text_searcher else None
            object_query = queries.object_detection_searcher.query if queries.object_detection_searcher else None
            final_results = self.reranker.rerank(
                merged_results, text_query, object_query)

            paginated_results = self._paginate_results(
                final_results, page, per_page)

            logger.info(
                f"Search completed. Total results: {len(final_results)}, Current page: {page}")
            return paginated_results
        except Exception as e:
            logger.error(
                f"Error occurred during search: {str(e)}", exc_info=True)
            raise

    def _paginate_results(self, sorted_results: List[FrameMetadataModel], page: int, per_page: int) -> SearchResult:
        total_results = len(sorted_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paged_frames = sorted_results[start_idx:end_idx]

        return SearchResult(
            frames=paged_frames,
            total=total_results,
            page=page,
            has_more=end_idx < total_results
        )
