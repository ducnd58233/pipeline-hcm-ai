import asyncio
from typing import Dict, Any, List
from app.models import SearchResult, FrameMetadataModel
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.services.fusion.abstract_fusion import AbstractFusion
from app.services.reranker.abstract_reranker import AbstractReranker
from app.log import logger


class SearchService:
    def __init__(self,
                 searchers: Dict[str, AbstractSearcher],
                 fusion: AbstractFusion,
                 reranker: AbstractReranker):
        self.searchers = searchers
        self.fusion = fusion
        self.reranker = reranker

    async def search(self, queries: Dict[str, Any], weights: Dict[str, float], page: int = 1, per_page: int = 20) -> SearchResult:
        try:
            logger.info(
                f"Performing search for queries: {queries}, weights: {weights}, page: {page}, per_page: {per_page}")

            search_tasks = []
            for searcher_name, query in queries.items():
                if not query:
                    continue
                search_tasks.append((searcher_name, self.searchers[searcher_name].search(
                    query, page=page, per_page=page*per_page)))

            results = await asyncio.gather(*[task[1] for task in search_tasks])
            searcher_results = {task[0]: result for task,
                                result in zip(search_tasks, results)}

            merged_results = self.fusion.merge_results(
                searcher_results, weights)
            final_results = self.reranker.rerank(merged_results)

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
