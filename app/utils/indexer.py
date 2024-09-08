import numpy as np
import faiss
from app.log import logger

logger = logger.getChild(__name__)

class FaissIndexer:
    def __init__(self, index_path: str):
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")
        logger.info(f"Index total vectors: {self.index.ntotal}")
        logger.info(f"Index dimension: {self.index.d}")

    def search(self, query_vector: np.ndarray, k: int) -> tuple:
        return self.index.search(query_vector.reshape(1, -1), k)
