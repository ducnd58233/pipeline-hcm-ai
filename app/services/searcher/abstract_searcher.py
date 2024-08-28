from abc import ABC, abstractmethod
from typing import List, Any
from app.models import SearchResult


class AbstractSearcher(ABC):
    @abstractmethod
    async def search(self, query: Any, page: int, per_page: int) -> SearchResult:
        pass
