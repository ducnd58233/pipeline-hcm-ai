from typing import Dict, List, Optional
from app.services.reranker.abstract_reranker import AbstractReranker
from app.models import FrameMetadataModel, ObjectQuery
from sentence_transformers import SentenceTransformer, util
import torch


class SentenceBertReranker(AbstractReranker):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def rerank(self, merged_results: Dict[str, FrameMetadataModel], text_query: Optional[str], object_query: Optional[ObjectQuery]) -> List[FrameMetadataModel]:
        # Combine text and object queries
        combined_query = self._combine_queries(text_query, object_query)

        # Encode the combined query
        query_embedding = self.model.encode(
            combined_query, convert_to_tensor=True)

        # Prepare frame texts and their corresponding FrameMetadataModel objects
        frame_texts = []
        frame_objects = []
        for frame in merged_results.values():
            # Combine text from various sources in the frame
            frame_text = f"{frame.keyframe.video_path} {frame.keyframe.frame_path}"
            if frame.detection:
                frame_text += " " + \
                    " ".join(f"{obj}{pos}" for pos,
                             obj in frame.detection.objects.items())
            frame_texts.append(frame_text)
            frame_objects.append(frame)

        # Encode frame texts
        frame_embeddings = self.model.encode(
            frame_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(
            query_embedding, frame_embeddings)[0]

        # Sort results by cosine similarity
        sorted_results = sorted(
            zip(frame_objects, cosine_scores), key=lambda x: x[1], reverse=True)

        # Update scores and return sorted list
        for frame, score in sorted_results:
            frame.score.value = float(score)
            frame.score.details['sentence_bert'] = float(score)

        return [frame for frame, _ in sorted_results]

    def _combine_queries(self, text_query: Optional[str], object_query: Optional[ObjectQuery]) -> str:
        combined_query_parts = []

        if text_query:
            combined_query_parts.append(text_query)

        if object_query:
            object_query_str = " ".join(
                f"{pos[1]}{pos[0]}{obj}" for pos, obj in object_query.objects.items())
            combined_query_parts.append(object_query_str)

        return " ".join(combined_query_parts) if combined_query_parts else ""
