from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import Score, SearchResult, FrameMetadataModel, TextQuery
from app.utils.data_manager.frame_data_manager import frame_data_manager
from app.utils.query_vectorizer.text_vectorizer import TextQueryVectorizer
from app.utils.indexer import FaissIndexer
from typing import List, Dict
from app.log import logger
import numpy as np


class TextSearcher(AbstractSearcher):
    def __init__(self, vectorizer: TextQueryVectorizer, indexer: FaissIndexer):
        self.vectorizer = vectorizer
        self.indexer = indexer

    async def search(self, query: TextQuery, page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing text search with query: {query.query}")

        query_structure = await self.vectorizer.parse_query(query.query)

        all_results: Dict[int, Dict] = {}
        for part in query_structure:
            if isinstance(part, list):
                part_query = " ".join(part)
                similar_frames = await self.search_similar_frames(part_query, top_k=per_page*page)
                self.increase_scores(all_results, similar_frames)

        self.rescale_scores(all_results)

        sorted_results = sorted(all_results.values(),
                                key=lambda x: x['similarity'], reverse=True)

        start = (page - 1) * per_page
        end = start + per_page

        result_frames: List[FrameMetadataModel] = []
        for result in sorted_results[start:end]:
            frame = frame_data_manager.get_frame_by_index(
                result['frame_index'])
            if frame:
                frame.score = Score(value=float(result['similarity']), details={
                                    'text': float(result['similarity']), 'match_count': result['match_count']})
                result_frames.append(frame)

        return SearchResult(
            frames=result_frames,
            total=len(sorted_results),
            page=page,
            has_more=end < len(sorted_results)
        )

    async def search_similar_frames(self, query: str, top_k: int = 5) -> List[dict]:
        query_vector = await self.vectorizer.vectorize(query)

        distances, indices = self.indexer.search(query_vector, k=top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            similarity = 1 / (1 + distance)
            results.append({
                'frame_index': int(idx),
                'similarity': float(similarity)
            })

        return results

    def increase_scores(self, all_results: Dict[int, Dict], new_results: List[Dict]):
        for result in new_results:
            frame_index = result['frame_index']
            if frame_index in all_results:
                all_results[frame_index]['similarity'] += result['similarity']
                all_results[frame_index]['match_count'] += 1
            else:
                result['match_count'] = 1
                all_results[frame_index] = result

    def rescale_scores(self, all_results: Dict[int, Dict]):
        scores = [result['similarity'] for result in all_results.values()]
        min_score = min(scores)
        max_score = max(scores)

        if max_score > min_score:
            for result in all_results.values():
                result['similarity'] = (
                    result['similarity'] - min_score) / (max_score - min_score)
        else:
            for result in all_results.values():
                result['similarity'] = 1.0
