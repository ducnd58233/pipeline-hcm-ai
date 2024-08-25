from app.abstract_classes import AbstractRelevanceCalculator
from app.models import FrameMetadataModel
import nltk
from nltk.corpus import wordnet
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EnhancedRelevanceCalculator(AbstractRelevanceCalculator):
    def __init__(self, text_processor):
        self.text_processor = text_processor
        nltk.download('wordnet', quiet=True)

    def calculate_relevance(self, query: str, metadata: FrameMetadataModel, query_structure: List[Union[str, List[str]]]) -> float:
        if not query_structure:
            return self._calculate_simple_relevance(query, metadata)

        relevance_scores = []
        for part in query_structure:
            if isinstance(part, list):
                part_query = " ".join(part)
                part_score = self._calculate_simple_relevance(
                    part_query, metadata)
                relevance_scores.append(part_score)
            elif isinstance(part, str) and part.lower() in ["and", "or"]:
                relevance_scores.append(part.lower())
            else:
                part_score = self._calculate_simple_relevance(
                    str(part), metadata)
                relevance_scores.append(part_score)

        final_score = self._combine_scores(relevance_scores)
        return final_score

    def _calculate_simple_relevance(self, query: str, metadata: FrameMetadataModel) -> float:
        query_tokens = set(
            self.text_processor.tokenize_and_remove_stopwords(query))

        relevant_text = []
        if metadata.detection:
            relevant_text.extend(metadata.detection.objects.keys())

        metadata_tokens = set(
            self.text_processor.tokenize_and_remove_stopwords(' '.join(relevant_text)))

        exact_match_score = len(query_tokens.intersection(
            metadata_tokens)) / (len(query_tokens) + 1e-10)
        semantic_similarity_score = self.calculate_semantic_similarity(
            query_tokens, metadata_tokens)

        return 0.65 * exact_match_score + 0.35 * semantic_similarity_score

    def calculate_semantic_similarity(self, query_tokens, metadata_tokens):
        max_similarities = []
        for q_token in query_tokens:
            q_synsets = wordnet.synsets(q_token)
            if not q_synsets:
                continue
            q_synset = q_synsets[0]

            similarities = []
            for m_token in metadata_tokens:
                m_synsets = wordnet.synsets(m_token)
                if not m_synsets:
                    continue
                m_synset = m_synsets[0]
                similarity = q_synset.path_similarity(m_synset) or 0
                similarities.append(similarity)

            if similarities:
                max_similarities.append(max(similarities))

        return sum(max_similarities) / (len(query_tokens) + 1e-10)

    def _combine_scores(self, scores: List[Union[float, str]]) -> float:
        logger.debug(f"Combining scores: {scores}")
        if len(scores) == 1:
            return scores[0] if isinstance(scores[0], float) else 0.0

        final_score = 1.0
        current_score = 0.0
        operator = "and"

        for score in scores:
            if isinstance(score, str):
                operator = score
            else:
                if operator == "and":
                    current_score *= score
                elif operator == "or":
                    current_score = max(current_score, score)

            if operator == "and" and isinstance(score, float):
                final_score *= current_score
                current_score = 1.0

        final_score *= current_score
        
        logger.debug(f"Final combined score: {final_score}")
        return min(final_score, 1.0)
