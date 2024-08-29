from abc import ABC, abstractmethod
from typing import Union
from app.models import SearchResult, TextQuery, ObjectQuery


class AbstractSearcher(ABC):
    @abstractmethod
    async def search(self, query: Union[TextQuery, ObjectQuery], page: int, per_page: int) -> SearchResult:
        pass
