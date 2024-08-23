import os
import faiss
import json
import numpy as np
from typing import List
from config import Config
from app.utils.clip_utils import get_clip_model, encode_text
from app.services.nlp_service import preprocess_query
from app.models import FrameMetadataModel


class FAISSService:
    def __init__(self):
        self.index = faiss.read_index(Config.FAISS_BIN_PATH)
        self.clip_model, self.clip_tokenizer, _ = get_clip_model()
        with open(Config.KEYFRAMES_METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)
        self.id_to_index = {key: idx for idx,
                            key in enumerate(self.metadata.keys())}
        self.index_to_id = {idx: key for key, idx in self.id_to_index.items()}

    async def search(self, query, page=1, per_page=20, extra_k=50):
        if isinstance(query, str):
            query_vector = await self.preprocess_query(query)
        elif isinstance(query, dict):
            query_vector = await self._metadata_to_vector(query)
        else:
            raise ValueError("Invalid query type. Expected string or dict.")

        k = page * per_page + extra_k

        scores, indices = self.index.search(query_vector, k)

        start = (page - 1) * per_page
        end = start + per_page
        return await self._get_results(scores[0][start:end], indices[0][start:end])

    async def preprocess_query(self, query):
        processed_query = await preprocess_query(query)
        query_vector = await encode_text(self.clip_model, self.clip_tokenizer, processed_query)
        if query_vector.shape[1] != self.index.d:
            padded_vector = np.zeros((1, self.index.d), dtype=np.float32)
            padded_vector[:, :min(query_vector.shape[1], self.index.d)] = query_vector[:, :min(
                query_vector.shape[1], self.index.d)]
            query_vector = padded_vector
        return query_vector

    async def _metadata_to_vector(self, query_metadata):
        most_similar_frame = max(self.metadata.items(),
                                 key=lambda x: self._calculate_similarity(query_metadata, x[1]))[0]
        frame_index = self.id_to_index[most_similar_frame]
        return self.index.reconstruct(frame_index).reshape(1, -1)

    def _calculate_similarity(self, query_metadata, frame_metadata):
        similarity = 0
        if query_metadata.get('video_path') == frame_metadata.get('video_path'):
            similarity += 1
        if abs(query_metadata.get('timestamp', 0) - frame_metadata.get('timestamp', 0)) < 1:
            similarity += 1
        return similarity

    async def _get_results(self, scores, indices) -> List[FrameMetadataModel]:
        results = []
        for score, idx in zip(scores, indices):
            frame_id = self.index_to_id.get(idx)
            if frame_id in self.metadata:
                frame_data = self.metadata[frame_id]
                frame_metadata_model = FrameMetadataModel(
                    id=frame_id,
                    shot_index=frame_data['shot_index'],
                    frame_index=frame_data['frame_index'],
                    timestamp=frame_data['timestamp'],
                    video_path=os.path.join('videos', frame_data['video_path']),
                    frame_path=os.path.join('keyframes', frame_data['frame_path']),
                    score=float(score),
                    selected=frame_data.get('selected', False)
                )
                results.append(frame_metadata_model)
        return results


faiss_service = FAISSService()
