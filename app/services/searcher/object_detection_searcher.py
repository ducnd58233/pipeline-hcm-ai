from typing import List, Dict
import numpy as np
from scipy.sparse import csr_matrix
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

        similar_frames = await self.search_similar_frames(query, top_k=per_page*page + 50)
        if not similar_frames:
            logger.warning("No similar frames found for the given query.")
            return SearchResult(frames=[], total=0, page=page, has_more=False)

        sorted_results = sorted(similar_frames,
                                key=lambda x: x['similarity'], reverse=True)
        result_frames = self.prepare_result_frames(
            sorted_results, page, per_page)

        total_results = len(sorted_results)
        logger.debug(f"Retrieved {total_results} frames")

        return SearchResult(
            frames=result_frames,
            total=total_results,
            page=page,
            has_more=len(result_frames) == per_page
        )

    async def search_similar_frames(self, query: ObjectQuery, top_k: int = 5) -> List[Dict[str, float]]:
        query_vector = self.vectorizer.vectorize(query)
        similarities, indices = self.vectorizer.search(query_vector, k=top_k)
        logger.debug(
            f"Search results - similarities: {similarities}, indices: {indices}")

        results = []
        for idx in indices:
            if similarities[idx] <= 0:
                continue

            results.append({
                'frame_index': int(idx),
                'similarity': float(similarities[idx])
            })

        return results
    
    def prepare_result_frames(self, sorted_results: List[Dict[str, float]], page: int, per_page: int) -> List[FrameMetadataModel]:
        start = (page - 1) * per_page
        end = start + per_page

        result_frames = []
        
        start = (page - 1) * per_page
        end = start + per_page

        result_frames: List[FrameMetadataModel] = []
        for result in sorted_results[start:end]:
            frame = frame_data_manager.get_frame_by_index(
                result['frame_index'])
            if frame:
                frame.score = Score(value=float(result['similarity']), details={
                                    'object': float(result['similarity'])})
                result_frames.append(frame)
            else:
                logger.warning(f"Frame not found for id: {result['frame_id']}")

        return result_frames
