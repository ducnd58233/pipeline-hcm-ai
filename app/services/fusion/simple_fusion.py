from .abstract_fusion import AbstractFusion
from app.models import SearchResult, FrameMetadataModel, Score, QueriesStructure
from typing import Dict
from app.log import logger
from app.utils.weight_normalizer import WeightNormalizer

logger = logger.getChild(__name__)


class SimpleFusion(AbstractFusion):
    def merge_results(self, searcher_results: Dict[str, SearchResult], queries: QueriesStructure) -> Dict[str, FrameMetadataModel]:
        merged = {}
        weights = {
            'text': queries.text_searcher.weight if queries.text_searcher else 0,
            'object': queries.object_detection_searcher.weight if queries.object_detection_searcher else 0,
            'tag': queries.text_searcher.weight if queries.text_searcher else 0,
        }

        weights = WeightNormalizer.normalize(weights)
        logger.info(f'Weights normalizer: {weights}')

        base_k = 60
        logger.debug(f'Simple fusion results before: {searcher_results}')
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
        
        if not merged:
            return {}

        for frame in merged.values():
            rrf_score = sum(
                weights[searcher] *
                (1 / (base_k + details['rank']) + details['score'])
                for searcher, details in frame.score.details.items()
                if weights[searcher] > 0
            )
            frame.score.value = rrf_score

        self.normalize_scores(merged)

        for frame in merged.values():
            logger.debug(f'Fusion frame detail: {frame}')

        return merged

    def normalize_scores(self, merged: Dict[str, FrameMetadataModel]):
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
