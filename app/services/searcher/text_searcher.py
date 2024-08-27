from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import Score, SearchResult
from app.utils.search_processor import TextProcessor
from app.utils.vectorizer import OpenClipVectorizer
from app.utils.indexer import FaissIndexer
from app.utils.frame_data_manager import frame_data_manager
import numpy as np
import logging


logger = logging.getLogger(__name__)

class TextSearcher(AbstractSearcher):
    def __init__(self, vectorizer: OpenClipVectorizer, indexer: FaissIndexer, text_processor: TextProcessor):
        self.vectorizer = vectorizer
        self.indexer = indexer
        self.text_processor = text_processor

    async def search(self, query: str, page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing object text search with query: {query}")
        offset = (page - 1) * per_page
        query_structure = await self.text_processor.parse_long_query(query)
        logger.info(f"Processed query: {query_structure}")

        all_results = []
        for part in query_structure:
            if isinstance(part, list):
                part_query = " ".join(part)
                similar_frames = self.search_similar_frames(
                    part_query, top_k=per_page+offset)
                frames = []
                for result in similar_frames[offset:]:
                    frame = frame_data_manager.get_frame_by_index(
                        result['frame_index'])
                    if frame:
                        frame.score = Score(
                            details={'text': float(result['similarity'])})
                        frames.append(frame)
                all_results.extend(frames)

        unique_results = {frame.id: frame for frame in all_results if frame}
        sorted_results = sorted(unique_results.values(),
                                key=lambda x: x.score.details.get('text', 0), reverse=True)

        return SearchResult(
            frames=sorted_results[:per_page],
            total=len(sorted_results),
            page=page,
            has_more=len(sorted_results) > offset + per_page
        )

    def search_similar_frames(self, query: str, top_k: int = 5) -> list:
        query_vector = self.vectorizer.vectorize(query).reshape(1, -1)
        vectors = self.indexer.index.reconstruct_n(
            0, self.indexer.index.ntotal)

        similarity = np.dot(vectors, query_vector.T).flatten()
        top_indices = similarity.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'frame_index': idx,
                'similarity': float(similarity[idx])
            })

        return results
