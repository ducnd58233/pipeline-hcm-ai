from .abstract_reranker import AbstractReranker
from app.models import FrameMetadataModel, ObjectQuery
from typing import Dict, List, Optional


class SimpleReranker(AbstractReranker):
    def rerank(self, merged_results: Dict[str, FrameMetadataModel], text_query: Optional[str], object_query: Optional[ObjectQuery]) -> List[FrameMetadataModel]:
        return sorted(merged_results.values(), key=lambda x: x.final_score, reverse=True)
