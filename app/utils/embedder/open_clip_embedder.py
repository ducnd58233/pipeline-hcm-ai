import numpy as np
import torch
import faiss
from typing import Optional, Tuple
from open_clip import create_model_and_transforms, get_tokenizer
from app.log import logger
from app.utils.indexer import FaissIndexer

from .abstract_embedder import AbstractTextEmbedder

logger = logger.getChild(__name__)

class OpenClipEmbedder(AbstractTextEmbedder):
    def __init__(self, model_name: str = 'ViT-L-14', pretrained: str = 'datacomp_xl_s13b_b90k', feature_shape: Optional[Tuple[int, ...]] = None):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model, _, _ = create_model_and_transforms(
            model_name, device=self.device, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = get_tokenizer(model_name)
        self.feature_shape = feature_shape

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        text_tokens = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        embedding = text_features.cpu().numpy()[0]
        logger.debug(f"Raw embedding shape: {embedding.shape}")
        logger.debug(f"Raw embedding: {embedding}")
        
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        embedding = embedding.reshape(-1)
        
        resized_embedding = self.resize_embedding(
            embedding, self.feature_shape)

        logger.debug(f"Resized embedding shape: {resized_embedding.shape}")
        logger.debug(f"Resized embedding: {resized_embedding}")

        return resized_embedding
