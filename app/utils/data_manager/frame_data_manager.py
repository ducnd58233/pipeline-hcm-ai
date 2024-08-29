import csv
import json
import os
from typing import Dict, List, Optional, Tuple
from app.models import FrameMetadataModel, KeyframeInfo
from app.services.redis_service import redis_service
from .visual_encoding_manager import VisualEncodingManager
from config import Config
import faiss
from app.log import logger
import numpy as np
from .object_detection_manager import ObjectDetectionManager


class FrameDataManager:
    def __init__(self, metadata_path: str, faiss_index_path: str):
        self.frame_data: Dict[str, FrameMetadataModel] = {}
        self.frame_id_to_index: Dict[str, int] = {}
        self.index_to_frame_id: Dict[int, str] = {}
        self.storage = redis_service
        self.metadata_path = metadata_path
        self.faiss_index_path = faiss_index_path

        classes = self._load_classes(
            f'{metadata_path}/indexing/metadata_encoded/classes.csv')
        visual_encoding_manager = VisualEncodingManager(classes)
        self.object_detection_manager = ObjectDetectionManager(
            visual_encoding_manager)

        self._load_frame_data()
        self.faiss_index = self._load_faiss_index()

    def _load_classes(self, csv_path: str) -> List[str]:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [row[0] for row in reader]

    def _load_frame_data(self):
        keyframes_data, object_detection_data = self._load_json_data()
        for idx, (frame_id, frame_info) in enumerate(keyframes_data.items()):
            frame_metadata = self._create_frame_metadata(
                frame_id, frame_info, object_detection_data)
            self._add_frame_to_data(idx, frame_id, frame_metadata)

        logger.info(f"Processed {len(self.frame_data)} frames")

    def _load_json_data(self) -> Tuple[Dict, Dict]:
        keyframes_path = os.path.join(
            self.metadata_path, 'keyframes_metadata.json')
        object_detection_path = os.path.join(
            self.metadata_path, 'object_extraction_metadata.json')
        with open(keyframes_path, 'r') as f:
            keyframes_data = json.load(f)
        with open(object_detection_path, 'r') as f:
            object_detection_data = json.load(f)
        return keyframes_data, object_detection_data
    
    def __get_frame_dimensions(self, frame_path: str) -> Tuple[int, int]:
        # full_path = os.path.join(Config.KEYFRAMES_DIR, frame_path)
        # img = cv2.imread(full_path)
        # if img is None:
        #     logger.warning(f"Could not read image: {full_path}")
        #     return 1280, 720
        # return img.shape[1], img.shape[0]
        return 1280, 720

    def _create_frame_metadata(self, frame_id: str, frame_info: Dict, object_detection_data: Dict) -> FrameMetadataModel:
        keyframe = KeyframeInfo(**frame_info)
        frame_dimensions = self.__get_frame_dimensions(
            f"keyframes/{keyframe.frame_path}")
        detection = self.object_detection_manager.process_object_detection(
            frame_id, object_detection_data, frame_dimensions)
        frame_metadata = FrameMetadataModel(
            id=frame_id, keyframe=keyframe, detection=detection)
        frame_metadata.keyframe.frame_path = frame_metadata.get_corrected_frame_path()
        frame_metadata.keyframe.video_path = frame_metadata.get_corrected_video_path()
        return frame_metadata
    
    def _add_frame_to_data(self, idx: int, frame_id: str, frame_metadata: FrameMetadataModel):
        self.frame_data[frame_id] = frame_metadata
        self.frame_id_to_index[frame_id] = idx
        self.index_to_frame_id[idx] = frame_id

    def _load_faiss_index(self):
        return faiss.read_index(self.faiss_index_path)

    def get_frame_by_id(self, frame_id: str) -> Optional[FrameMetadataModel]:
        frame = self.frame_data.get(frame_id)
        selected_frame_key = self.__get_selected_frames_key()
        score = self.__get_selected_frames_score_key()
        if frame:
            frame.selected = redis_service.is_member_of_set(
                selected_frame_key, frame_id)
            frame.final_score = redis_service.zscore(
                score, frame_id) or 0.0
        return frame

    def get_frame_by_index(self, index: int) -> Optional[FrameMetadataModel]:
        frame_id = self.index_to_frame_id.get(index)
        if frame_id:
            return self.get_frame_by_id(frame_id)
        return None

    def get_all_frames(self) -> List[FrameMetadataModel]:
        return list(self.frame_data.values())

    def search_similar_frames(self, query_vector: np.ndarray, k: int = 10) -> List[FrameMetadataModel]:
        _, indices = self.faiss_index.search(query_vector.reshape(1, -1), k)
        similar_frames = []
        for idx in indices[0]:
            frame = self.get_frame_by_index(int(idx))
            if frame:
                similar_frames.append(frame)
        return similar_frames

    def toggle_frame_selection(self, frame_id: str, score: float = 0.0) -> bool:
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
            redis_service.zadd(score_key, {frame_id: score})
            frame.selected = True
            frame.final_score = score

        return frame.selected

    def get_selected_frames(self) -> List[FrameMetadataModel]:
        selected_frame_key = self.__get_selected_frames_key()
        selected_frame_ids = redis_service.get_set_members(selected_frame_key)
        return [self.get_frame_by_id(frame_id) for frame_id in selected_frame_ids if frame_id in self.frame_data]

    def clear_all(self):
        selected_frame_key = self.__get_selected_frames_key()
        score_key = self.__get_selected_frames_score_key()

        redis_service.delete_key(selected_frame_key)
        redis_service.delete_key(score_key)

        for frame in self.frame_data.values():
            frame.selected = False
            frame.final_score = 0.0
            frame.score.details = {}

        logger.info(
            f"Cleared all selected frames and scores for user {Config.USER_ID}")

    def __get_selected_frames_key(self):
        return f"selected_frames:{Config.USER_ID}"

    def __get_selected_frames_score_key(self):
        return f"frame_scores:{Config.USER_ID}"

frame_data_manager = FrameDataManager(
    Config.METADATA_PATH, Config.FAISS_BIN_PATH)
