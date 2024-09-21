from typing import List, Tuple
from app.utils.query_vectorizer.abstract_query_vectorizer import AbstractQueryVectorizer
from app.utils.search_processor import TextProcessor
from app.utils.embedder.abstract_embedder import AbstractTextEmbedder
from app.log import logger
import numpy as np
import faiss

logger = logger.getChild(__name__)


class TextQueryVectorizer(AbstractQueryVectorizer):
    def __init__(self, embedder: AbstractTextEmbedder, text_processor: TextProcessor, faiss_index: faiss.Index):
        self.embedder = embedder
        self.text_processor = text_processor
        self.faiss_index = faiss_index

    async def vectorize(self, query: str) -> Tuple[np.ndarray, List[str]]:
        preprocessed_query = await self.parse_query(query)
        logger.info(
            f'Text processed query: {preprocessed_query}')

        embedding = self.embedder.embed(preprocessed_query)

        logger.debug(f"Final query vector shape: {embedding.shape}")
        logger.debug(f"Final query vector: {embedding}")

        return embedding, entities

    async def preprocess_query(self, query: str) -> str:
        return await self.text_processor.preprocess_query(query)

    async def parse_query(self, query: str) -> Tuple[str, List[str]]:
        preprocessed_query = await self.preprocess_query(query)
        return preprocessed_query

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.faiss_index.search(query_vector, k)

        return distances.flatten(), indices.flatten()

