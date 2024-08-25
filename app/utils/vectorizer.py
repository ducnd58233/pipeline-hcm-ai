import numpy as np
from open_clip import create_model_and_transforms, tokenize
import torch
from app.abstract_classes import AbstractVectorizer


class ClipVectorizer(AbstractVectorizer):
    def __init__(self, model_name="ViT-L-14"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = create_model_and_transforms(
            model_name, device=device)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def vectorize(self, text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            return text_features.cpu().numpy()
