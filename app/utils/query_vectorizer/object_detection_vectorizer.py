import os
import pickle
from scipy.sparse import load_npz

from sklearn.metrics.pairwise import cosine_similarity
from app.utils.query_vectorizer.abstract_query_vectorizer import AbstractQueryVectorizer
from app.models import ObjectQuery
import numpy as np
from app.log import logger

from config import Config

logger = logger.getChild(__name__)

class ObjectQueryVectorizer(AbstractQueryVectorizer):
    def __init__(self):
        self.vectorizer = self.__load_vectorizer()
        self.vectors = self.__load_vectors()
        logger.debug(f'vectorizer: {self.vectorizer}')
        logger.debug(f'vector: {self.vectors}')
        

    def __load_vectorizer(self):
        vectorizer_path = os.path.join(
            Config.OD_ENCODED_DIR, 'bbox_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def __load_vectors(self):
        vectors_path = os.path.join(
            Config.OD_ENCODED_DIR, 'bbox_vectors.npz')
        return load_npz(vectors_path)

    def vectorize(self, query: ObjectQuery) -> np.ndarray:
        parsed_query = query.parse_query()
        query_text = ' '.join(parsed_query)
        logger.info(f'Object processed query: {query_text}')
        return self.vectorizer.transform([query_text])

    def search(self, query_vector, k):   
        similarities = cosine_similarity(self.vectors, query_vector).flatten()
                
        top_indices = similarities.argsort()[-k:][::-1]
        return similarities, top_indices
