from typing import List
from app.models import FrameMetadataModel
from app.abstract_classes import AbstractVectorizer, AbstractIndexer, AbstractReranker
from app.utils.frame_data_manager import frame_data_manager


class SearchService:
    def __init__(self, vectorizer: AbstractVectorizer, indexer: AbstractIndexer, reranker: AbstractReranker):
        self.vectorizer = vectorizer
        self.indexer = indexer
        self.reranker = reranker
        self.batch_size = 100
        self.max_results = 1000

    async def search(self, query: str, offset: int = 0, limit: int = 20) -> List[FrameMetadataModel]:
        query_vector = self.vectorizer.vectorize(query)
        all_results = []
        current_offset = 0

        while len(all_results) < offset + limit and len(all_results) < self.max_results:
            batch_size = min(
                self.batch_size, self.max_results - len(all_results))
            scores, indices = self.indexer.search(
                query_vector, current_offset + batch_size)

            batch_results = self.get_frame_metadata(indices[0], scores[0])
            reranked_batch = self.reranker.rerank(query, batch_results)
            all_results.extend(reranked_batch)

            if len(batch_results) < batch_size:
                break

            current_offset += batch_size

        return all_results[offset:offset+limit]

    def get_frame_metadata(self, indices, scores) -> List[FrameMetadataModel]:
        results = []
        for idx, score in zip(indices, scores):
            frame = frame_data_manager.get_frame_by_index(int(idx))
            if frame:
                frame.score = score
                results.append(frame)
        return results
