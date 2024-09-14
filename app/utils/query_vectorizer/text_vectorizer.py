from typing import List, Tuple
from app.utils.query_vectorizer.abstract_query_vectorizer import AbstractQueryVectorizer
from app.utils.search_processor import TextProcessor
from app.utils.embedder.abstract_embedder import AbstractTextEmbedder
from app.log import logger
import numpy as np

logger = logger.getChild(__name__)


class TextQueryVectorizer(AbstractQueryVectorizer):
    def __init__(self, embedder: AbstractTextEmbedder, text_processor: TextProcessor):
        self.embedder = embedder
        self.text_processor = text_processor

    async def vectorize(self, query: str) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        preprocessed_query, entities = await self.parse_query(query)
        logger.info(
            f'Text processed query: {preprocessed_query} - entities: {entities}')

        embedding = self.embedder.embed(preprocessed_query)

        logger.debug(f"Final query vector shape: {embedding.shape}")
        logger.debug(f"Final query vector: {embedding}")

        return embedding, entities

    async def preprocess_query(self, query: str) -> str:
        return await self.text_processor.preprocess_query(query)

    async def parse_query(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        return await self.text_processor.parse_query(query)
