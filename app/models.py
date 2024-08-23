import os
import logging
import json
from app.error import FrameNotFoundError
from config import Config
from typing import Dict, Optional
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)



class FrameMetadataModel(BaseModel):
    id: str
    shot_index: int
    frame_index: int
    timestamp: float
    video_path: str
    frame_path: str
    score: Optional[float] = Field(default=0.0)
    selected: Optional[bool] = Field(default=False)


class FrameMetadata:
    _metadata: Dict[str, FrameMetadataModel] = {}
    _index_to_id: Dict[int, str] = {}

    @classmethod
    async def load_metadata(cls):
        if not cls._metadata:
            logger.info(
                f"Loading metadata from {Config.KEYFRAMES_METADATA_PATH}")
            try:
                with open(Config.KEYFRAMES_METADATA_PATH, 'r') as f:
                    raw_metadata = json.load(f)
                    cls._metadata = {
                        frame_id: FrameMetadataModel(
                            id=frame_id,
                            shot_index=data['shot_index'],
                            frame_index=data['frame_index'],
                            timestamp=data['timestamp'],
                            video_path=os.path.join(
                                'videos', data['video_path']),
                            frame_path=os.path.join(
                                'keyframes', data['frame_path']),
                            score=data.get('score'),
                            selected=data.get('selected')
                        )
                        for frame_id, data in raw_metadata.items()
                    }
                    cls._index_to_id = {idx: frame_id for idx,
                                        frame_id in enumerate(cls._metadata.keys())}
                logger.info(f"Loaded metadata for {len(cls._metadata)} frames")
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                raise

    @classmethod
    async def get_by_frame_id(cls, frame_id: str) -> FrameMetadataModel:
        await cls.load_metadata()
        frame_data = cls._metadata.get(frame_id)
        if frame_data:
            return frame_data
        raise FrameNotFoundError(f"Frame not found: {frame_id}")

    @classmethod
    async def get_by_index(cls, index: int) -> FrameMetadataModel:
        await cls.load_metadata()
        frame_id = cls._index_to_id.get(index)
        if frame_id:
            return await cls.get_by_frame_id(frame_id)
        raise FrameNotFoundError(f"Frame not found for index: {index}")

    @staticmethod
    async def get_total_frames() -> int:
        await FrameMetadata.load_metadata()
        return len(FrameMetadata._metadata)
