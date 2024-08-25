import numpy as np
import faiss
from app.abstract_classes import AbstractIndexer

class FaissIndexer(AbstractIndexer):
    def __init__(self, index_path: str):
        self.index = faiss.read_index(index_path)

    def search(self, query_vector: np.ndarray, k: int) -> tuple:
        return self.index.search(query_vector.reshape(1, -1), k)
