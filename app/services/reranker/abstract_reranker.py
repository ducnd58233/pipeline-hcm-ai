from abc import ABC, abstractmethod
from typing import Dict, List
from app.models import FrameMetadataModel


class AbstractReranker(ABC):
    @abstractmethod
    def rerank(self, merged_results: Dict[str, FrameMetadataModel]) -> List[FrameMetadataModel]:
        pass
