from typing import List
import asyncio
from app.models import FrameMetadataModel
from app.abstract_classes import AbstractVectorizer, AbstractIndexer, AbstractReranker
from app.utils.frame_data_manager import frame_data_manager


class SearchService:
    def __init__(self, vectorizer: AbstractVectorizer, indexer: AbstractIndexer, reranker: AbstractReranker, text_processor):
        self.vectorizer = vectorizer
        self.indexer = indexer
        self.reranker = reranker
        self.text_processor = text_processor
        self.batch_size = 100
        self.max_results = 1000

    async def search(self, query: str, offset: int = 0, limit: int = 200) -> List[FrameMetadataModel]:
        query_structure = await self.text_processor.parse_long_query(query)

        all_results = []
        search_tasks = []

        for query_part in query_structure:
            if isinstance(query_part, list):
                sub_query = " ".join(query_part)
            elif isinstance(query_part, str):
                sub_query = query_part
            else:
                continue

            query_vector = self.vectorizer.vectorize(sub_query)
            search_tasks.append(self._search_part(query_vector))

        results = await asyncio.gather(*search_tasks)
        for part_results in results:
            all_results.extend(part_results)

        unique_results = list(
            {result.id: result for result in all_results}.values())
        reranked_results = self.reranker.rerank(
            query, unique_results, query_structure)

        return reranked_results[offset:offset+limit]

    async def _search_part(self, query_vector) -> List[FrameMetadataModel]:
        results = []
        current_offset = 0

        while len(results) < self.max_results:
            batch_size = min(self.batch_size, self.max_results - len(results))
            scores, indices = self.indexer.search(
                query_vector, current_offset + batch_size)

            batch_results = self.get_frame_metadata(indices[0], scores[0])
            results.extend(batch_results)

            if len(batch_results) < batch_size:
                break

            current_offset += batch_size

        return results

    def get_frame_metadata(self, indices, scores) -> List[FrameMetadataModel]:
        results = []
        for idx, score in zip(indices, scores):
            frame = frame_data_manager.get_frame_by_index(int(idx))
            if frame:
                frame.score = score
                results.append(frame)
        return results
