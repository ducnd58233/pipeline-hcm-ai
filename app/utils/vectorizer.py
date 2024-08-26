import numpy as np
import torch
from app.abstract_classes import AbstractVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging
from open_clip import create_model_and_transforms, get_tokenizer

logger = logging.getLogger(__name__)


class OpenClipVectorizer(AbstractVectorizer):
    def __init__(self, model_name='ViT-L-14', pretrained='datacomp_xl_s13b_b90k', feature_shape=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = create_model_and_transforms(
            model_name, device=self.device, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = get_tokenizer(model_name)
        self.feature_shape = feature_shape

    @torch.no_grad()
    def vectorize(self, text: str) -> np.ndarray:
        text_tokens = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        vector = text_features.cpu().numpy()[0]
        return self.resize_vector(vector)

    def resize_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.feature_shape is None or vector.shape == self.feature_shape:
            return vector

        logger.warning(
            f"Resizing vector from {vector.shape} to {self.feature_shape}")
        resized_vector = np.zeros(self.feature_shape)
        min_shape = [min(s1, s2)
                     for s1, s2 in zip(vector.shape, self.feature_shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        resized_vector[slices] = vector[slices]

        return normalize(resized_vector.reshape(1, -1))[0]

class SentenceTransformerVectorizer(AbstractVectorizer):
    def __init__(self, model_name='all-MiniLM-L6-v2', feature_shape=None):
        self.model = SentenceTransformer(model_name)
        self.feature_shape = feature_shape

    @torch.no_grad()
    def vectorize(self, text: str) -> np.ndarray:
        vector = self.model.encode([text])[0]
        return self.resize_vector(vector)

    def resize_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.feature_shape is None or vector.shape == self.feature_shape:
            return vector

        logger.warning(
            f"Resizing vector from {vector.shape} to {self.feature_shape}")
        resized_vector = np.zeros(self.feature_shape)
        min_shape = [min(s1, s2)
                     for s1, s2 in zip(vector.shape, self.feature_shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        resized_vector[slices] = vector[slices]

        return normalize(resized_vector.reshape(1, -1))[0]
