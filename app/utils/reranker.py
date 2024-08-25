from typing import List
from app.abstract_classes import AbstractRelevanceCalculator, AbstractReranker
from app.models import FrameMetadataModel

class EnhancedReranker(AbstractReranker):
    def __init__(self, relevance_calculator: AbstractRelevanceCalculator):
        self.relevance_calculator = relevance_calculator
        self.weights = {
            'clip_score': 0.7,
            'relevance_score': 0.3
        }

    def rerank(self, query: str, results: List[FrameMetadataModel]) -> List[FrameMetadataModel]:
        for result in results:
            relevance_score = self.relevance_calculator.calculate_relevance(
                query, result)
            result.final_score = (
                self.weights['clip_score'] * result.score +
                self.weights['relevance_score'] * relevance_score
            )
        return sorted(results, key=lambda x: x.final_score, reverse=True)

