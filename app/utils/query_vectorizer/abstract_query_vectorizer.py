from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractQueryVectorizer(ABC):
    @abstractmethod
    def vectorize(self, query: Any) -> np.ndarray:
        pass
