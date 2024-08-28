from app.services.fusion.abstract_fusion import AbstractFusion
from app.models import SearchResult, FrameMetadataModel, Score
from typing import Dict
from app.log import logger

logger = logger.getChild(__name__)

class SimpleFusion(AbstractFusion):
    def merge_results(self, searcher_results: Dict[str, SearchResult], weights: Dict[str, float]) -> Dict[str, FrameMetadataModel]:
        merged = {}
        for searcher_name, result in searcher_results.items():
            for frame in result.frames:
                if frame.id not in merged:
                    merged[frame.id] = frame.model_copy(deep=True)
                    merged[frame.id].score = Score(value=0.0, details={})
                merged[frame.id].score.details[searcher_name] = frame.score.value
                weight = weights.get(searcher_name, 0)
                merged[frame.id].score.value += weight * frame.score.value

        max_score = max(
            (frame.score.value for frame in merged.values()), default=0)
        if max_score > 0:
            for frame in merged.values():
                frame.score.value /= max_score
                frame.final_score = frame.score.value

        for frame in merged.values():
            logger.info(f'Fusion frame detail: {frame}')


        return merged
