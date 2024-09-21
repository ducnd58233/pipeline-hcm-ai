import os
import pickle
from typing import List, Tuple
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from app.log import logger

from app.models import TagQuery
from app.utils.query_vectorizer.abstract_query_vectorizer import AbstractQueryVectorizer
from app.utils.search_processor import TextProcessor
from config import Config

logger = logger.getChild(__name__)


class TagQueryVectorizer(AbstractQueryVectorizer):
    def __init__(self, text_processor: TextProcessor, tags_list: List[str]):
        self.vectorizer = self.__load_vectorizer()
        self.vectors = self.__load_vectors()
        self.text_processor = text_processor
        self.tags_list = tags_list

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

    async def vectorize(self, query: TagQuery) -> Tuple[np.ndarray, List[str]]:
        query_terms = await self.text_processor.extract_relevant_terms(query.query)
        
        all_terms = list(set(query_terms + query.entities))
        existed_terms = [term for term in all_terms if term in self.tags_list]
        
        logger.info(
            f'All terms for vectorization: {existed_terms}')
        term_vectors = self.vectorizer.transform([" ".join(existed_terms)])
        return term_vectors, existed_terms

    def search(self, query_vector, k):
        if query_vector.shape[0] > 1:
            similarities = cosine_similarity(
                self.vectors, query_vector).max(axis=1).flatten()
        else:
            similarities = cosine_similarity(
                self.vectors, query_vector).flatten()

        top_indices = similarities.argsort()[-k:][::-1]
        return similarities, top_indices
