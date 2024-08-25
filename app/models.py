from typing import Optional, Dict, List
import os
import logging
import json
from app.error import FrameNotFoundError
from config import Config
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)


class ObjectDetectionItem(BaseModel):
    score: float
    box: List[float]


class ObjectDetection(BaseModel):
    objects: Dict[str, List[ObjectDetectionItem]]
    counts: Dict[str, int]


class KeyframeInfo(BaseModel):
    shot_index: int
    frame_index: int
    shot_start: int
    shot_end: int
    shot_start: int
    shot_end: int
    timestamp: float
    video_path: str
    frame_path: str


class FrameMetadataModel(BaseModel):
    id: str
    keyframe: KeyframeInfo
    detection: Optional[ObjectDetection] = None
    score: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    selected: bool = Field(default=False)
    
    def get_corrected_frame_path(self):
        return f"keyframes/{self.keyframe.frame_path}"
    
    def get_corrected_video_path(self):
        return f"videos/{self.keyframe.video_path}"
