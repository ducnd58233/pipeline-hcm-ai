import json
from config import Config
import logging

logger = logging.getLogger(__name__)


class FrameMetadata:
    _metadata = None
    _index_to_id = None

    def __init__(self, frame_id, shot_index, frame_index, timestamp, video_path, frame_path):
        self.id = frame_id
        self.shot_index = shot_index
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.video_path = video_path
        self.frame_path = frame_path

    @classmethod
    async def load_metadata(cls):
        if cls._metadata is None:
            logger.info(
                f"Loading metadata from {Config.KEYFRAMES_METADATA_PATH}")
            with open(Config.KEYFRAMES_METADATA_PATH, 'r') as f:
                cls._metadata = json.load(f)
            cls._index_to_id = {idx: frame_id for idx,
                                frame_id in enumerate(cls._metadata.keys())}
            logger.info(f"Loaded metadata for {len(cls._metadata)} frames")

    @classmethod
    async def get_by_id(cls, frame_id):
        await cls.load_metadata()
        data = cls._metadata.get(frame_id)
        if data:
            return cls(
                frame_id=frame_id,
                shot_index=data['shot_index'],
                frame_index=data['frame_index'],
                timestamp=data['timestamp'],
                video_path=data['video_path'],
                frame_path=data['frame_path']
            )
        return None

    @classmethod
    async def get_by_index(cls, index):
        await cls.load_metadata()
        frame_id = cls._index_to_id.get(index)
        if frame_id:
            return await cls.get_by_id(frame_id)
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'shot_index': self.shot_index,
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            'video_path': self.video_path,
            'frame_path': self.frame_path
        }

    @staticmethod
    async def get_total_frames():
        await FrameMetadata.load_metadata()
        return len(FrameMetadata._metadata)

    def __repr__(self):
        return f"<FrameMetadata id={self.id} frame_index={self.frame_index}>"
