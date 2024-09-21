import numpy as np
from typing import List, Tuple
from app.log import logger
from app.utils.indexer import FaissIndexer

logger = logger.getChild(__name__)


class GlobalReranker:
    def __init__(self, frame_data_manager, faiss_index: FaissIndexer):
        self.frame_data_manager = frame_data_manager
        self.faiss_index = faiss_index

    def initial_retrieval(self, text_query_embedding: np.ndarray, num_initial_results: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        distances, frame_indices = self.faiss_index.search(
            text_query_embedding.reshape(1, -1), num_initial_results)
        # Convert distances to similarities
        initial_similarities = 1 / (1 + distances[0])

        valid_indices = frame_indices[0]
        valid_similarities = initial_similarities

        # Retrieve embeddings directly from FAISS
        initial_embeddings = self.get_embeddings_from_faiss(valid_indices)

        return valid_indices, valid_similarities, initial_embeddings

    def get_embeddings_from_faiss(self, indices: np.ndarray) -> np.ndarray:
        embeddings = np.empty(
            (len(indices), self.faiss_index.index.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            self.faiss_index.index.reconstruct(int(idx), embeddings[i])
        return embeddings

    def refine_embeddings(self, initial_embeddings: np.ndarray, num_neighbors: int = 5, similarity_weight: float = 0.15) -> np.ndarray:
        refined_embeddings = []
        for embedding in initial_embeddings:
            neighbor_distances, neighbor_indices = self.faiss_index.search(
                embedding.reshape(1, -1), num_neighbors)
            neighbor_similarities = 1 / (1 + neighbor_distances[0])
            neighbor_embeddings = self.get_embeddings_from_faiss(
                neighbor_indices[0])
            refined_embedding = self.weighted_average_pooling(
                neighbor_embeddings, neighbor_similarities, similarity_weight)
            refined_embeddings.append(refined_embedding)
        return np.array(refined_embeddings)

    def weighted_average_pooling(self, embeddings: np.ndarray, similarities: np.ndarray, similarity_weight: float = 0.15) -> np.ndarray:
        weights = similarities ** similarity_weight
        weighted_sum = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        return weighted_sum / np.sum(weights)

    def expand_query(self, text_query_embedding: np.ndarray, num_neighbors: int = 5) -> np.ndarray:
        _, query_neighbor_indices = self.faiss_index.search(
            text_query_embedding.reshape(1, -1), num_neighbors)
        query_neighbor_embeddings = self.get_embeddings_from_faiss(
            query_neighbor_indices[0])
        return np.max(query_neighbor_embeddings, axis=0)

    def compute_final_scores(self, text_query_embedding: np.ndarray, refined_embeddings: np.ndarray, initial_embeddings: np.ndarray, expanded_query: np.ndarray) -> np.ndarray:
        similarity_to_refined = np.dot(
            refined_embeddings, text_query_embedding.T).flatten()
        similarity_to_expanded = np.dot(
            initial_embeddings, expanded_query.T).flatten()
        return (similarity_to_refined + similarity_to_expanded) / 2

    def rerank(self, text_query_embedding: np.ndarray, num_initial_results: int = 1000, num_neighbors: int = 5, similarity_weight: float = 0.15) -> List[Tuple[str, float]]:
        # Initial retrieval
        initial_frame_indices, _, initial_embeddings = self.initial_retrieval(
            text_query_embedding, num_initial_results)

        if len(initial_frame_indices) == 0:
            logger.warning("No results found in initial retrieval")
            return []

        # Refine embeddings
        refined_embeddings = self.refine_embeddings(
            initial_embeddings, num_neighbors, similarity_weight)

        # Expand query
        expanded_query = self.expand_query(
            text_query_embedding, num_neighbors)

        # Compute final scores
        final_scores = self.compute_final_scores(
            text_query_embedding, refined_embeddings, initial_embeddings, expanded_query)

        # Rerank
        reranked_indices = np.argsort(final_scores)[::-1]

        # Get frame IDs and scores
        reranked_results = []
        for idx in reranked_indices:
            frame = self.frame_data_manager.get_frame_by_index(
                int(initial_frame_indices[idx]))
            if frame:
                reranked_results.append((frame.id, float(final_scores[idx])))

        return reranked_results
