from app.abstract_classes import AbstractRelevanceCalculator
from app.models import FrameMetadataModel


class EnhancedRelevanceCalculator(AbstractRelevanceCalculator):
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def calculate_relevance(self, query: str, metadata: FrameMetadataModel) -> float:
        query_tokens = set(
            self.text_processor.tokenize_and_remove_stopwords(query))

        relevant_text = []
        if metadata.detection:
            relevant_text.extend(metadata.detection.objects.keys())

        metadata_tokens = set(
            self.text_processor.tokenize_and_remove_stopwords(' '.join(relevant_text)))

        common_tokens = query_tokens.intersection(metadata_tokens)
        return len(common_tokens) / (len(query_tokens) + 1e-10)

