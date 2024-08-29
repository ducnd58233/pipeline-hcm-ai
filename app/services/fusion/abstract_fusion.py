from abc import ABC, abstractmethod
from typing import Dict
from app.models import SearchResult, FrameMetadataModel, QueriesStructure


class AbstractFusion(ABC):
    @abstractmethod
    def merge_results(self, searcher_results: Dict[str, SearchResult], queries: QueriesStructure) -> Dict[str, FrameMetadataModel]:
        pass
