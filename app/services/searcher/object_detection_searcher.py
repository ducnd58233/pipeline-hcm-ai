from typing import List
from app.log import logger
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import FrameMetadataModel, ObjectQuery, Score, SearchResult
from app.utils.frame_data_manager import frame_data_manager
from app.utils.query_vectorizer.object_detection_vectorizer import ObjectQueryVectorizer
import numpy as np

logger = logger.getChild(__name__)


class ObjectDetectionSearcher(AbstractSearcher):
    def __init__(self, vectorizer: ObjectQueryVectorizer):
        self.vectorizer = vectorizer

    async def search(self, query: ObjectQuery, page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing object detection search with query: {query}")

        similar_frames = await self.search_similar_frames(query, top_k=per_page*page)

        start = (page - 1) * per_page
        end = start + per_page

        result_frames: List[FrameMetadataModel] = []
        for result in similar_frames[start:end]:
            frame = frame_data_manager.get_frame_by_index(result['frame_index'])
            if frame:
                frame.score = Score(value=float(result['similarity']), details={
                                    'object': float(result['similarity'])})
                result_frames.append(frame)

        return SearchResult(
            frames=result_frames,
            total=len(similar_frames),
            page=page,
            has_more=end < len(similar_frames)
        )

    async def search_similar_frames(self, query: ObjectQuery, top_k: int = 5) -> List[dict]:
        query_vector = self.vectorizer.vectorize(query)

        # Calculate cosine similarity
        similarities = self.vectorizer.vectors.dot(
            query_vector.T).toarray().flatten()

        # Normalize similarities to [0, 1] range
        similarities = (similarities - similarities.min()) / \
            (similarities.max() - similarities.min() + 1e-10)

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'frame_index': int(idx),
                'similarity': float(similarities[idx])
            })

        return results
