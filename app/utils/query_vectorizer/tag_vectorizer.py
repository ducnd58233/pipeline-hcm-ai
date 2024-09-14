import os
import pickle
from typing import List, Tuple
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from app.log import logger
from app.models import TextQuery
from app.utils.query_vectorizer.abstract_query_vectorizer import AbstractQueryVectorizer
from app.utils.search_processor import TextProcessor
from config import Config

logger = logger.getChild(__name__)


class TagQueryVectorizer(AbstractQueryVectorizer):
    def __init__(self, text_processor: TextProcessor):
        self.vectorizer = self.__load_vectorizer()
        self.vectors = self.__load_vectors()
        self.text_processor = text_processor
        logger.debug(f'vectorizer: {self.vectorizer}')
        logger.debug(f'vector: {self.vectors}')

    def __load_vectorizer(self):
        vectorizer_path = os.path.join(
            Config.TAG_ENCODED_DIR, 'multi_tag_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def __load_vectors(self):
        vectors_path = os.path.join(
            Config.TAG_ENCODED_DIR, 'multi_tag_vectors.npz'
        )
        return load_npz(vectors_path)

    async def vectorize(self, query: TextQuery) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        preprocessed_query, entities = await self.text_processor.parse_query(query.query)
        logger.info(
            f'Tag processed query: {preprocessed_query} - entities: {entities}')
        query_vector = self.vectorizer.transform([preprocessed_query])
        return query_vector, entities

    def search(self, query_vector, k):
        similarities = cosine_similarity(self.vectors, query_vector).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return similarities, top_indices
