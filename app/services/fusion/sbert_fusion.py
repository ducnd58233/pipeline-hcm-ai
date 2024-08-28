from typing import Dict
from sentence_transformers import SentenceTransformer, util
import torch
from app.services.fusion.abstract_fusion import AbstractFusion
from app.models import FrameMetadataModel, Score, SearchResult, QueriesStructure
from app.utils.weight_normalizer import WeightNormalizer


class SentenceBertFusion(AbstractFusion):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def merge_results(self, searcher_results: Dict[str, SearchResult], queries: QueriesStructure) -> Dict[str, FrameMetadataModel]:
        merged = {}
        weights = {
            'text': queries.text_searcher.weight if queries.text_searcher else 0,
            'object': queries.object_detection_searcher.weight if queries.object_detection_searcher else 0
        }
        normalized_weights = WeightNormalizer.normalize(weights)

        for searcher_name, result in searcher_results.items():
            for frame in result.frames:
                if frame.id not in merged:
                    merged[frame.id] = frame.model_copy(deep=True)
                    merged[frame.id].score = Score(value=0.0, details={})
                merged[frame.id].score.details[searcher_name] = frame.score.value * \
                    normalized_weights[searcher_name]

        combined_query = ""
        if queries.text_searcher:
            combined_query += queries.text_searcher.query.query + " "
        if queries.object_detection_searcher:
            combined_query += " ".join([f"{pos[1]}{pos[0]}{obj}" for pos,
                                       obj in queries.object_detection_searcher.query.objects.items()])

        query_vector = self.model.encode(
            combined_query, convert_to_tensor=True)
        frame_vectors = torch.stack([self.model.encode(
            frame.id, convert_to_tensor=True) for frame in merged.values()])
        similarities = util.pytorch_cos_sim(
            query_vector, frame_vectors).squeeze()

        for frame, similarity in zip(merged.values(), similarities):
            frame.score.details['sbert'] = similarity.item()
            frame.final_score = sum(
                frame.score.details.values()) / len(frame.score.details)

        return merged
