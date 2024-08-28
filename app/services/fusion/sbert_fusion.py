from typing import Dict
from sentence_transformers import SentenceTransformer, util
import torch
from app.services.fusion.abstract_fusion import AbstractFusion
from app.models import FrameMetadataModel, Score, SearchResult


class SBERTFusion(AbstractFusion):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def merge_results(self, searcher_results: Dict[str, SearchResult], query_vector: torch.Tensor) -> Dict[str, FrameMetadataModel]:
        merged = {}
        for searcher_name, result in searcher_results.items():
            for frame in result.frames:
                if frame.id not in merged:
                    merged[frame.id] = frame.model_copy(deep=True)
                    merged[frame.id].score = Score(value=0.0, details={})
                merged[frame.id].score.details[searcher_name] = frame.score.value

        frame_vectors = torch.stack([self.model.encode(
            frame.id, convert_to_tensor=True) for frame in merged.values()])
        similarities = util.pytorch_cos_sim(
            query_vector, frame_vectors).squeeze()

        for frame, similarity in zip(merged.values(), similarities):
            frame.score.details['sbert'] = similarity.item()
            frame.final_score = sum(
                frame.score.details.values()) / len(frame.score.details)

        return merged
    