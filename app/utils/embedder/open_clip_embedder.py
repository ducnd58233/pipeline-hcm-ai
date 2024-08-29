import numpy as np
import torch
from typing import Optional, Tuple
from open_clip import create_model_and_transforms, get_tokenizer

from .abstract_embedder import AbstractTextEmbedder


class OpenClipEmbedder(AbstractTextEmbedder):
    def __init__(self, model_name: str = 'ViT-L-14', pretrained: str = 'datacomp_xl_s13b_b90k', feature_shape: Optional[Tuple[int, ...]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        return self.resize_embedding(embedding, self.feature_shape)
