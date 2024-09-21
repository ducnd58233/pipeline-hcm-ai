from typing import List, Dict, Tuple, Optional
import numpy as np
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import Score, SearchResult, FrameMetadataModel, TagQuery
from app.utils.data_manager.frame_data_manager import frame_data_manager
from app.utils.query_vectorizer.tag_vectorizer import TagQueryVectorizer
from app.log import logger

logger = logger.getChild(__name__)


class TagSearcher(AbstractSearcher):
    def __init__(self, vectorizer: TagQueryVectorizer):
        self.vectorizer = vectorizer

    async def search(self, query: TagQuery, page: int, per_page: int, boost_factors: Optional[Dict[str, float]] = None) -> SearchResult:
        logger.info(
            f"Performing tag search with query: {query.query}, additional entities: {query.entities}")

        query_vector, terms = await self.vectorizer.vectorize(query)
        similar_frames = await self.search_similar_frames(query_vector, top_k=per_page*page + 50)

        if not similar_frames:
            logger.warning("No similar frames found for the given query.")
            return SearchResult(frames=[], total=0, page=page, has_more=False)

        self.apply_boost_and_normalize(similar_frames, boost_factors)

        sorted_results = sorted(
            similar_frames, key=lambda x: x['final_score'], reverse=True)
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

    async def search_similar_frames(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, float]]:
        similarities, indices = self.vectorizer.search(query_vector, k=top_k)

        results = []
        for idx in indices:
            if similarities[idx] <= 0:
                continue

            frame_id = frame_data_manager.index_to_frame_key.get(int(idx))
            if frame_id:
                results.append({
                    'frame_id': frame_id,
                    'similarity': float(similarities[idx])
                })
            else:
                logger.warning(f"No frame_id found for index {idx}")

        return results

    def apply_boost_and_normalize(self, frames: List[Dict[str, float]], boost_factors: Optional[Dict[str, float]] = None):
        if boost_factors is None:
            boost_factors = {frame['frame_id']: 1.0 for frame in frames}

        for frame in frames:
            frame['boost'] = boost_factors.get(frame['frame_id'], 1.0)
            frame['final_score'] = frame['similarity'] * frame['boost']

        total_score = sum(frame['final_score'] for frame in frames)
        if total_score > 1.0:
            normalization_factor = 1.0 / total_score
            for frame in frames:
                frame['final_score'] *= normalization_factor

    def prepare_result_frames(self, sorted_results: List[Dict[str, float]], page: int, per_page: int) -> List[FrameMetadataModel]:
        start = (page - 1) * per_page
        end = start + per_page

        result_frames = []
        for result in sorted_results[start:end]:
            frame = frame_data_manager.get_frame_by_key(result['frame_id'])
            if frame:
                frame.score = Score(value=float(result['final_score']), details={
                    'tag': float(result['similarity']),
                    'boost': float(result['boost'])
                })
                result_frames.append(frame)
            else:
                logger.warning(f"Frame not found for id: {result['frame_id']}")

        return result_frames
