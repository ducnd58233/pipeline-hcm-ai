from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from app.models import FrameMetadataModel, ObjectQuery


class AbstractReranker(ABC):
    @abstractmethod
    def rerank(self, merged_results: Dict[str, FrameMetadataModel], text_query: Optional[str], object_query: Optional[ObjectQuery]) -> List[FrameMetadataModel]:
        pass
