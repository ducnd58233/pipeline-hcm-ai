from typing import Dict
from app.models import SearchResult, FrameMetadataModel, Score


class LateFusionSearcher:
    def merge_results(self, searcher_results: Dict[str, SearchResult]) -> Dict[str, FrameMetadataModel]:
        merged = {}
        for searcher_name, result in searcher_results.items():
            for frame in result.frames:
                if frame.id not in merged:
                    merged[frame.id] = frame.model_copy(deep=True)
                    merged[frame.id].score = Score(value=0.0, details={})
                merged[frame.id].score.details[searcher_name] = frame.score.value
        return merged
