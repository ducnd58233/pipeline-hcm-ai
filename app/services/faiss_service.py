import faiss
import json
import os
from config import Config
from app.utils.clip_utils import get_clip_model, encode_text
from app.services.nlp_service import preprocess_query


class FAISSService:
    def __init__(self):
        self.index = faiss.read_index(Config.FAISS_BIN_PATH)
        self.clip_model = get_clip_model()
        with open(Config.KEYFRAMES_METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)
        self.id_to_index = {key: idx for idx,
                            key in enumerate(self.metadata.keys())}
        self.index_to_id = {idx: key for key, idx in self.id_to_index.items()}

    def search(self, query, k=100):
        if isinstance(query, str):
            query_vector = preprocess_query(query, expected_dim=self.index.d)
        elif isinstance(query, dict):
            query_vector = self._metadata_to_vector(query)
        else:
            raise ValueError("Invalid query type. Expected string or dict.")
        
        # Perform the search
        scores, indices = self.index.search(query_vector, k)
        return self._get_results(scores[0], indices[0])

    def _metadata_to_vector(self, query_metadata):
        # Find the most similar frame based on metadata
        most_similar_frame = None
        max_similarity = -1

        for frame_id, metadata in self.metadata.items():
            similarity = self._calculate_similarity(query_metadata, metadata)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_frame = frame_id

        if most_similar_frame is None:
            raise ValueError("No matching frame found for the given metadata")

        # Get the index of the most similar frame
        frame_index = self.id_to_index[most_similar_frame]

        # Reconstruct the vector for this frame
        return self.index.reconstruct(frame_index).reshape(1, -1)

    def _calculate_similarity(self, query_metadata, frame_metadata):
        # Implement a simple similarity metric based on metadata fields
        # You can customize this based on your specific metadata structure
        similarity = 0
        if query_metadata.get('video_path') == frame_metadata.get('video_path'):
            similarity += 1
        if abs(query_metadata.get('timestamp', 0) - frame_metadata.get('timestamp', 0)) < 1:
            similarity += 1
        return similarity

    def _get_results(self, scores, indices):
        results = []
        for score, idx in zip(scores, indices):
            frame_id = self.index_to_id.get(idx)
            if frame_id in self.metadata:
                results.append({
                    'frame_id': frame_id,
                    'score': float(score),
                    'metadata': self.metadata[frame_id]
                })
        return results
