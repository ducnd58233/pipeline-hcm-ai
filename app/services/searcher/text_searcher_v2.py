import numpy as np
from app.services.reranker.global_reranker import GlobalReranker
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import Score, SearchResult, FrameMetadataModel, TextQuery
from app.utils.data_manager.frame_data_manager import frame_data_manager
from app.utils.query_vectorizer.text_vectorizer import TextQueryVectorizer
from typing import Dict, List, Tuple
from app.log import logger

logger = logger.getChild(__name__)


class TextSearcherV2(AbstractSearcher):
    def __init__(self, vectorizer: TextQueryVectorizer):
        self.vectorizer = vectorizer
        self.reranker = GlobalReranker(
            frame_data_manager, vectorizer.faiss_index)

    async def search(self, query: TextQuery, page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing text search with query: {query.query}")

        query_embedding = await self.vectorizer.vectorize(query.query)
        num_initial_results = per_page * page + 100
        reranked_results = self.reranker.rerank(
            query_embedding, num_initial_results=num_initial_results)

        if not reranked_results:
            logger.warning(
                "No results from reranker, falling back to simple search")
            reranked_results = self.fallback_search(
                query_embedding, num_initial_results)

        if not reranked_results:
            logger.warning("No results found even after fallback")
            return SearchResult(frames=[], total=0, page=page, has_more=False)

        paginated_frames: List[FrameMetadataModel] = self.prepare_result_frames(
            reranked_results, page, per_page)

        total_results = len(reranked_results)
        logger.info(f"Retrieved {total_results} frames")

        return SearchResult(
            frames=paginated_frames,
            total=total_results,
            page=page,
            has_more=len(paginated_frames) == per_page
        )

    def fallback_search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        distances, indices = self.vectorizer.faiss_index.search(
            query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            frame = frame_data_manager.get_frame_by_index(int(idx))
            if frame:
                similarity = 1 / (1 + distance)
                results.append((frame.id, float(similarity)))
        return results

    def prepare_result_frames(self, reranked_results: List[Tuple[str, float]], page: int, per_page: int) -> List[FrameMetadataModel]:
        start = (page - 1) * per_page
        end = start + per_page

        paginated_frames: List[FrameMetadataModel] = []
        for frame_id, similarity_score in reranked_results[start:end]:
            frame = frame_data_manager.get_frame_by_key(frame_id)
            if frame:
                frame.score = Score(value=float(similarity_score), details={
                                    'text': float(similarity_score)})
                paginated_frames.append(frame)

        return paginated_frames
