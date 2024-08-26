import asyncio
import logging
from typing import List, Dict, Any
from app.models import SearchResult, FrameMetadataModel, Score
from app.services.searcher.abstract_searcher import AbstractSearcher

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, searchers: Dict[str, AbstractSearcher], weights: Dict[str, float]):
        self.searchers = searchers
        self.weights = weights

    async def search(self, text_query: str, object_query: Dict[str, Any] = None, page: int = 1, per_page: int = 20) -> SearchResult:
        try:
            logger.info(
                f"Performing combined search for text query: {text_query}, object_query: {object_query}, page: {page}, per_page: {per_page}")

            search_tasks = []
            for searcher_name, searcher in self.searchers.items():
                if searcher_name == 'object' and object_query:
                    search_tasks.append((searcher_name, searcher.search(
                        object_query, page=1, per_page=per_page*page)))
                if searcher_name == 'text' and text_query:
                    search_tasks.append((searcher_name, searcher.search(
                        text_query, page=1, per_page=per_page*page)))

            results = await asyncio.gather(*[task[1] for task in search_tasks])
            searcher_results = {task[0]: result for task,
                                result in zip(search_tasks, results)}

            merged_results = self.__merge_results(searcher_results)
            final_results = self.__apply_weights_and_calculate_final_score(
                merged_results)
            paginated_results = self.__paginate_results(
                final_results, page, per_page)

            logger.info(
                f"Search completed. Total results: {len(final_results)}, Current page: {page}")
            return paginated_results
        except Exception as e:
            logger.error(
                f"Error occurred during search: {str(e)}", exc_info=True)
            raise

    def __merge_results(self, searcher_results: Dict[str, SearchResult]) -> Dict[str, FrameMetadataModel]:
        merged = {}
        for searcher_name, result in searcher_results.items():
            for frame in result.frames:
                if frame.id not in merged:
                    merged[frame.id] = frame.model_copy(deep=True)
                merged[frame.id].score.details.update(frame.score.details)
        return merged

    def __apply_weights_and_calculate_final_score(self, merged_results: Dict[str, FrameMetadataModel]) -> List[FrameMetadataModel]:
        for frame in merged_results.values():
            frame.score.value = sum(self.weights.get(searcher, 0.0) * score
                                    for searcher, score in frame.score.details.items())

        return sorted(merged_results.values(), key=lambda x: x.score.value, reverse=True)

    def __paginate_results(self, sorted_results: List[FrameMetadataModel], page: int, per_page: int) -> SearchResult:
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paged_frames = sorted_results[start_idx:end_idx]

        return SearchResult(
            frames=paged_frames,
            total=len(sorted_results),
            page=page,
            has_more=end_idx < len(sorted_results)
        )
