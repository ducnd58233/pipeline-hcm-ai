from abc import ABC, abstractmethod
from typing import List, Any
from app.models import FrameMetadataModel


class AbstractVectorizer(ABC):
    @abstractmethod
    def vectorize(self, text: str) -> Any:
        pass


class AbstractIndexer(ABC):
    @abstractmethod
    def search(self, query_vector: Any, k: int) -> tuple:
        pass


class AbstractRelevanceCalculator(ABC):
    @abstractmethod
    def calculate_relevance(self, query: str, metadata: FrameMetadataModel) -> float:
        pass


class AbstractReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[FrameMetadataModel]) -> List[FrameMetadataModel]:
        pass
