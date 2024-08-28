from abc import ABC, abstractmethod
from typing import Dict
from app.models import SearchResult, FrameMetadataModel


class AbstractFusion(ABC):
    @abstractmethod
    def merge_results(self, searcher_results: Dict[str, SearchResult], weights: Dict[str, float]) -> Dict[str, FrameMetadataModel]:
        pass
