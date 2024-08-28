from typing import Dict, List
from app.models import FrameMetadataModel, Score


class Reranker:
    def rerank(self, merged_results: Dict[str, FrameMetadataModel], weights: Dict[str, float]) -> List[FrameMetadataModel]:
        for frame in merged_results.values():
            weighted_score = sum(weights.get(searcher, 0.0) * score
                                 for searcher, score in frame.score.details.items())
            frame.final_score = weighted_score

        return sorted(merged_results.values(), key=lambda x: x.score.value, reverse=True)
