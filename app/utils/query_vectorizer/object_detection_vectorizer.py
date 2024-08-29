import os
import pickle
from scipy.sparse import load_npz
from typing import List
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

    def __load_vectorizer(self):
        vectorizer_path = os.path.join(
            Config.METADATA_DIR, 'bbox_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def __load_vectors(self):
        vectors_path = os.path.join(
            Config.METADATA_DIR, 'bbox_vectors.npz')
        return load_npz(vectors_path)

    def vectorize(self, query: ObjectQuery) -> np.ndarray:
        parsed_query = query.parse_query()
        query_text = ' '.join(parsed_query)
        logger.info(f'Object processed query: {query_text}')
        return self.vectorizer.transform([query_text])

