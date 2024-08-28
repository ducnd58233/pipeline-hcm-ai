import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize
from app.log import logger

logger = logger.getChild(__name__)


class AbstractTextEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

    def resize_embedding(self, embedding: np.ndarray, target_shape: Optional[Tuple[int, ...]]) -> np.ndarray:
        if target_shape is None or embedding.shape == target_shape:
            return embedding

        logger.warning(
            f"Resizing embedding from {embedding.shape} to {target_shape}")
        resized_embedding = np.zeros(target_shape)
        min_shape = [min(s1, s2)
                     for s1, s2 in zip(embedding.shape, target_shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        resized_embedding[slices] = embedding[slices]

        return normalize(resized_embedding.reshape(1, -1))[0]
