from app.services.reranker.abstract_reranker import AbstractReranker
from app.models import FrameMetadataModel
from sentence_transformers import util
from typing import Dict, List
import torch


class SBERTReranker(AbstractReranker):
    def __init__(self, model):
        self.model = model

    def rerank(self, merged_results: Dict[str, FrameMetadataModel], query_vector: torch.Tensor) -> List[FrameMetadataModel]:
        frame_vectors = torch.stack([self.model.encode(
            frame.id, convert_to_tensor=True) for frame in merged_results.values()])
        similarities = util.pytorch_cos_sim(
            query_vector, frame_vectors).squeeze()

        for frame, similarity in zip(merged_results.values(), similarities):
            frame.final_score = similarity.item()

        return sorted(merged_results.values(), key=lambda x: x.final_score, reverse=True)
