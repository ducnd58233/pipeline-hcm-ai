from typing import Optional, Dict, List
from pydantic import BaseModel, Field


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
    timestamp: float
    video_path: str
    frame_path: str


class Score(BaseModel):
    value: float = 0.0
    details: Dict[str, float] = Field(default_factory=dict)


class FrameMetadataModel(BaseModel):
    id: str
    keyframe: KeyframeInfo
    detection: Optional[ObjectDetection] = None
    score: Score = Field(default_factory=Score)
    selected: bool = Field(default=False)

    def get_corrected_frame_path(self):
        return f"keyframes/{self.keyframe.frame_path}"

    def get_corrected_video_path(self):
        return f"videos/{self.keyframe.video_path}"

    @property
    def final_score(self):
        return self.score.value

    @final_score.setter
    def final_score(self, value):
        self.score.value = value


class SearchResult(BaseModel):
    frames: List[FrameMetadataModel]
    total: int
    page: int
    has_more: bool
