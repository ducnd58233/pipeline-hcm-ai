from .abstract_fusion import AbstractFusion
from app.models import SearchResult, FrameMetadataModel, Score, QueriesStructure
from typing import Dict, List
from app.log import logger
from app.utils.weight_normalizer import WeightNormalizer

logger = logger.getChild(__name__)


class SimpleFusion(AbstractFusion):
    def merge_results(self, searcher_results: Dict[str, SearchResult], queries: QueriesStructure, use_tag_inference: bool) -> Dict[str, FrameMetadataModel]:
        merged = self._collect_frames(searcher_results)

        if not merged:
            logger.warning("No results to merge.")
            return {}

        weights = self._calculate_weights(queries, use_tag_inference)
        self._calculate_scores(merged, weights)
        self._normalize_scores(merged)
        return merged

    def _collect_frames(self, searcher_results: Dict[str, SearchResult]) -> Dict[str, FrameMetadataModel]:
        merged = {}
        for searcher_name, result in searcher_results.items():
            for rank, frame in enumerate(result.frames, start=1):
                idx = frame.keyframe.frame_index
                if idx not in merged:
                    merged[idx] = frame.model_copy(deep=True)
                    merged[idx].score = Score(value=0.0, details={})
                merged[idx].score.details[searcher_name] = {
                    'rank': rank,
                    'score': frame.score.value if frame.score else 0.0
                }
        return merged

    def _calculate_weights(self, queries: QueriesStructure, use_tag_inference: bool) -> Dict[str, float]:
        weights = {
            'text': queries.text_searcher.weight if queries.text_searcher else 0,
            'object': queries.object_detection_searcher.weight if queries.object_detection_searcher else 0,
            'tag': queries.tag_searcher.weight if queries.tag_searcher else 0,
            'tag_inference': queries.text_searcher.weight if queries.text_searcher and use_tag_inference else 0,
        }
        return WeightNormalizer.normalize(weights)

    def _calculate_scores(self, merged: Dict[str, FrameMetadataModel], weights: Dict[str, float]):
        base_k = 60
        for frame in merged.values():
            rrf_score = sum(
                weights[searcher] *
                (1 / (base_k + details['rank']) + details['score'])
                for searcher, details in frame.score.details.items()
                if weights.get(searcher, 0) > 0
            )
            frame.score.value = rrf_score

    def _normalize_scores(self, merged: Dict[str, FrameMetadataModel]):
        if not merged:
            logger.warning("No frames to normalize scores.")
            return

        scores = [frame.score.value for frame in merged.values()]
        min_score = min(scores)
        max_score = max(scores)

        if max_score > min_score:
            for frame in merged.values():
                frame.score.value = (frame.score.value -
                                     min_score) / (max_score - min_score)
        else:
            for frame in merged.values():
                frame.score.value = 1.0

        for frame in merged.values():
            frame.final_score = frame.score.value
