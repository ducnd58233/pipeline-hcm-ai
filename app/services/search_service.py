import asyncio
from typing import List
from app.models import SearchResult, FrameMetadataModel, QueriesStructure
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.services.fusion.abstract_fusion import AbstractFusion
from app.services.reranker.abstract_reranker import AbstractReranker
from app.log import logger

logger = logger.getChild(__name__)

class SearchService:
    def __init__(self,
                 text_searcher: AbstractSearcher,
                 object_detection_searcher: AbstractSearcher,
                 fusion: AbstractFusion,
                 reranker: AbstractReranker):
        self.text_searcher = text_searcher
        self.object_detection_searcher = object_detection_searcher
        self.fusion = fusion
        self.reranker = reranker

    async def search(self, queries: QueriesStructure, page: int = 1, per_page: int = 20) -> SearchResult:
        try:
            logger.info(
                f"Performing search for queries: {queries}, page: {page}, per_page: {per_page}")

            search_tasks = []
            if queries.text_searcher:
                search_tasks.append(self.text_searcher.search(
                    queries.text_searcher.query, page=page, per_page=page*per_page))

            if queries.object_detection_searcher:
                search_tasks.append(self.object_detection_searcher.search(
                    queries.object_detection_searcher.query, page=page, per_page=page*per_page))

            results = await asyncio.gather(*search_tasks)

            searcher_results = {}
            if queries.text_searcher:
                searcher_results['text'] = results[0]
            if queries.object_detection_searcher:
                searcher_results['object'] = results[-1]

            merged_results = self.fusion.merge_results(
                searcher_results, queries)

            text_query = queries.text_searcher.query.query if queries.text_searcher else None
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
