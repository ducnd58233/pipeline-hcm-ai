import asyncio
from typing import List, Dict, Any
from app.log import logger
from app.models import SearchResult, FrameMetadataModel
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.services.late_fusion import LateFusionSearcher
from app.services.reranker import Reranker

logger = logger.getChild(__name__)


class SearchService:
    def __init__(self, searchers: Dict[str, AbstractSearcher]):
        self.searchers = searchers
        self.late_fusion = LateFusionSearcher()
        self.reranker = Reranker()

    async def search(self, queries: Dict[str, Any], weights: Dict[str, float], page: int = 1, per_page: int = 20) -> SearchResult:
        try:
            logger.info(
                f"Performing late fusion search for queries: {queries}, weights: {weights}, page: {page}, per_page: {per_page}")

            active_searchers = {searcher for searcher,
                                query in queries.items() if query}
            adjusted_weights = self._adjust_weights(weights, active_searchers)

            logger.info(
                f"Active searchers: {active_searchers}, Adjusted weights: {adjusted_weights}")

            search_tasks = []
            for searcher_name in active_searchers:
                search_tasks.append((searcher_name, self.searchers[searcher_name].search(
                    queries[searcher_name], page=1, per_page=per_page*page)))

            results = await asyncio.gather(*[task[1] for task in search_tasks])
            searcher_results = {task[0]: result for task,
                                result in zip(search_tasks, results)}

            merged_results = self.late_fusion.merge_results(searcher_results)
            final_results = self.reranker.rerank(
                merged_results, adjusted_weights)
            paginated_results = self._paginate_results(
                final_results, page, per_page)

            logger.info(
                f"Search completed. Total results: {len(final_results)}, Current page: {page}")
            return paginated_results
        except Exception as e:
            logger.error(
                f"Error occurred during search: {str(e)}", exc_info=True)
            raise

    def _adjust_weights(self, original_weights: Dict[str, float], active_searchers: set) -> Dict[str, float]:
        active_weights = {
            k: v for k, v in original_weights.items() if k in active_searchers}

        if not active_weights:
            return {searcher: 1.0 / len(active_searchers) for searcher in active_searchers}

        weight_sum = sum(active_weights.values())

        if weight_sum == 0:
            return {searcher: 1.0 / len(active_searchers) for searcher in active_searchers}

        return {k: v / weight_sum for k, v in active_weights.items()}

    def _paginate_results(self, sorted_results: List[FrameMetadataModel], page: int, per_page: int) -> SearchResult:
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paged_frames = sorted_results[start_idx:end_idx]

        return SearchResult(
            frames=paged_frames,
            total=len(sorted_results),
            page=page,
            has_more=end_idx < len(sorted_results)
        )
