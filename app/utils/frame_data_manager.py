import json
from typing import Dict, List, Optional
from app.models import FrameMetadataModel, KeyframeInfo, ObjectDetection
from app.services.redis_service import redis_service
from config import Config
import faiss
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FrameDataManager:
    def __init__(self, json_path: str, faiss_index_path: str):
        self.frame_data: Dict[str, FrameMetadataModel] = {}
        self.frame_id_to_index: Dict[str, int] = {}
        self.index_to_frame_id: Dict[int, str] = {}
        self.storage = redis_service
        self.__load_frame_data(json_path)
        self.faiss_index = self.__load_faiss_index(faiss_index_path)

    def __load_frame_data(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)

        for idx, (frame_id, frame_info) in enumerate(data.items()):
            keyframe = KeyframeInfo(**frame_info['keyframe'])
            detection = ObjectDetection(
                **frame_info['detection']) if 'detection' in frame_info else None
            frame_metadata = FrameMetadataModel(
                id=frame_id,
                keyframe=keyframe,
                detection=detection
            )
            frame_metadata.keyframe.frame_path = frame_metadata.get_corrected_frame_path()
            frame_metadata.keyframe.video_path = frame_metadata.get_corrected_video_path()
            self.frame_data[frame_id] = frame_metadata
            self.frame_id_to_index[frame_id] = idx
            self.index_to_frame_id[idx] = frame_id

    def __load_faiss_index(self, faiss_index_path: str):
        return faiss.read_index(faiss_index_path)

    def get_frame_by_id(self, frame_id: str) -> Optional[FrameMetadataModel]:
        frame = self.frame_data.get(frame_id)
        selected_frame_key = self.__get_selected_frames_key()
        score = self.__get_selected_frames_score_key()
        if frame:
            frame.selected = redis_service.is_member_of_set(
                selected_frame_key, frame_id)
            frame.score = redis_service.zscore(
                score, frame_id) or 0.0
        return frame

    def get_frame_by_index(self, index: int) -> Optional[FrameMetadataModel]:
        frame_id = self.index_to_frame_id.get(index)
        if frame_id:
            return self.get_frame_by_id(frame_id)
        return None

    def search_similar_frames(self, query_vector: np.ndarray, k: int = 10) -> List[FrameMetadataModel]:
        _, indices = self.faiss_index.search(query_vector.reshape(1, -1), k)
        similar_frames = []
        for idx in indices[0]:
            frame = self.get_frame_by_index(int(idx))
            if frame:
                similar_frames.append(frame)
        return similar_frames

    def toggle_frame_selection(self, frame_id: str, final_score: float = 0.0) -> bool:
        frame = self.get_frame_by_id(frame_id)
        if not frame:
            return False

        selected_frame_key = self.__get_selected_frames_key()
        score_key = self.__get_selected_frames_score_key()

        if frame.selected:
            redis_service.remove_from_set(selected_frame_key, frame_id)
            redis_service.zrem(score_key, frame_id)
            frame.selected = False
            frame.final_score = 0.0
        else:
            redis_service.add_to_set(selected_frame_key, frame_id)
            redis_service.zadd(score_key, {frame_id: final_score})
            frame.selected = True
            frame.final_score = final_score

        return frame.selected

    def get_selected_frames(self) -> List[FrameMetadataModel]:
        selected_frame_key = self.__get_selected_frames_key()
        selected_frame_ids = redis_service.get_set_members(selected_frame_key)
        return [self.get_frame_by_id(frame_id) for frame_id in selected_frame_ids if frame_id in self.frame_data]

    def clear_all(self):
        """Clear all selected frames and reset related states."""
        selected_frame_key = self.__get_selected_frames_key()
        score_key = self.__get_selected_frames_score_key()

        redis_service.delete_key(selected_frame_key)
        redis_service.delete_key(score_key)

        for frame in self.frame_data.values():
            frame.selected = False
            frame.final_score = 0.0
            frame.score = 0.0

        logger.info(
            f"Cleared all selected frames and scores for user {Config.USER_ID}")

    def __get_selected_frames_key(self):
        return f"selected_frames:{Config.USER_ID}"

    def __get_selected_frames_score_key(self):
        return f"frame_scores:{Config.USER_ID}"


frame_data_manager = FrameDataManager(
    Config.METADATA_PATH, Config.FAISS_BIN_PATH)
