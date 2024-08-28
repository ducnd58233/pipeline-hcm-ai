from app.services.reranker.abstract_reranker import AbstractReranker
from app.models import FrameMetadataModel
from typing import Dict, List


class SimpleReranker(AbstractReranker):
    def rerank(self, merged_results: Dict[str, FrameMetadataModel]) -> List[FrameMetadataModel]:
        return sorted(merged_results.values(), key=lambda x: x.final_score, reverse=True)
