import numpy as np
import torch
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer

from .abstract_embedder import AbstractTextEmbedder


class SentenceTransformerEmbedder(AbstractTextEmbedder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', feature_shape: Optional[Tuple[int, ...]] = None):
        self.model = SentenceTransformer(model_name)
        self.feature_shape = feature_shape

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text])[0]
        return self.resize_embedding(embedding, self.feature_shape)
