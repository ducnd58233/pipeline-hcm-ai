import os
import pickle
from scipy.sparse import load_npz
from typing import Dict, Any
import logging
from app.services.searcher.abstract_searcher import AbstractSearcher
from app.models import SearchResult
from app.utils.frame_data_manager import frame_data_manager
from config import Config

logger = logging.getLogger(__name__)


class ObjectDetectionSearcher(AbstractSearcher):
    def __init__(self):
        self.vectorizer = self.__load_vectorizer()
        self.vectors = self.__load_vectors()

    def __load_vectorizer(self):
        vectorizer_path = os.path.join(
            Config.METADATA_DIR, 'object_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def __load_vectors(self):
        vectors_path = os.path.join(
            Config.METADATA_DIR, 'object_vectors.npz')
        return load_npz(vectors_path)

    async def search(self, query: Dict[str, Any], page: int, per_page: int) -> SearchResult:
        logger.info(f"Performing object detection search with query: {query}")

        query_text = ' '.join(
            [f"{obj} " * count for obj, count in query.items()])
        query_vector = self.vectorizer.transform([query_text])

        similarities = self.vectors.dot(query_vector.T).toarray().flatten()
        sorted_indices = similarities.argsort()[::-1]

        start = (page - 1) * per_page
        end = start + per_page

        result_indices = sorted_indices[start:end]
        result_similarities = similarities[result_indices]

        frames = []
        for idx, similarity in zip(result_indices, result_similarities):
            frame_id = self.frame_ids[idx]
            frame = await frame_data_manager.get_frame_by_id(frame_id)
            if frame:
                frame.score = float(similarity)
                frames.append(frame)

        total_results = len(sorted_indices)

        return SearchResult(
            frames=frames,
            total=total_results,
            page=page,
            has_more=end < total_results
        )
