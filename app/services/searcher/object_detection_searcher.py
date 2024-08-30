from typing import List
import numpy as np
from app.log import logger
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import FrameMetadataModel, ObjectQuery, Score, SearchResult
from app.utils.data_manager.frame_data_manager import frame_data_manager
from app.utils.query_vectorizer.object_detection_vectorizer import ObjectQueryVectorizer

logger = logger.getChild(__name__)

class ObjectDetectionSearcher(AbstractSearcher):
    def __init__(self, vectorizer: ObjectQueryVectorizer):
        self.vectorizer = vectorizer

    async def search(self, query: ObjectQuery, page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing object detection search with query: {query}")

        all_similar_frames = await self.search_similar_frames(query)

        start = (page - 1) * per_page
        end = start + per_page
        paginated_frames = all_similar_frames[start:end]

        result_frames: List[FrameMetadataModel] = []
        for result in paginated_frames:
            frame = frame_data_manager.get_frame_by_index(
                result['frame_index'])
            if frame:
                frame.score = Score(value=float(result['similarity']), details={
                                    'object': float(result['similarity'])})
                result_frames.append(frame)

        total_results = len(all_similar_frames)
        has_more = end < total_results

        return SearchResult(
            frames=result_frames,
            total=total_results,
            page=page,
            has_more=has_more
        )

    async def search_similar_frames(self, query: ObjectQuery) -> List[dict]:
        query_vector = self.vectorizer.vectorize(query)

        similarities = self.vectorizer.vectors.dot(
            query_vector.T).toarray().flatten()

        exp_similarities = np.exp(similarities - np.max(similarities))
        softmax_similarities = exp_similarities / exp_similarities.sum()

        sorted_indices = np.argsort(softmax_similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if softmax_similarities[idx] > 0:
                results.append({
                    'frame_index': int(idx),
                    'similarity': float(softmax_similarities[idx])
                })

        return results
