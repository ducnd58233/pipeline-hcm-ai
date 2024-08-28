# models.py

from typing import Optional, Dict, List, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class QueryLogic(str, Enum):
    AND = "and"
    OR = "or"


class Category(str, Enum):
    AIRPLANE = "airplane"
    BICYCLE = "bicycle"
    BIRD = "bird"
    BOAT = "boat"
    CAT = "cat"
    DOG = "dog"
    PERSON = "person"


class ObjectQuery(BaseModel):
    objects: Dict[str, Category] = Field(default_factory=dict)
    logic: QueryLogic = QueryLogic.AND
    max_objects: Optional[int] = None

    @classmethod
    def from_query_params(cls, objects: Dict[str, str], logic: str, max_objects: Optional[str] = None) -> 'ObjectQuery':
        parsed_objects = {k: Category(v) for k, v in objects.items()}
        return cls(
            objects=parsed_objects,
            logic=QueryLogic(logic),
            max_objects=int(max_objects) if max_objects is not None else None
        )

    @validator('max_objects')
    def check_max_objects(cls, v):
        if v is not None and v <= 0:
            raise ValueError('max_objects must be a positive integer')
        return v


class SearchRequest(BaseModel):
    text_query: Optional[str] = None
    objects_query: ObjectQuery
    logic: QueryLogic = QueryLogic.AND
    max_objects: Optional[int] = None
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)

    @classmethod
    def from_form(cls, form_data: Dict[str, str]) -> 'SearchRequest':
        objects = {}
        for key, value in form_data.items():
            if key.startswith("objects.") and value:
                row, col = key.split(".")[1:]
                objects[f"{row},{col}"] = Category(value)

        return cls(
            text_query=form_data.get("text_query"),
            objects_query=ObjectQuery(objects=objects),
            logic=QueryLogic(form_data.get("logic", "and")),
            max_objects=int(form_data["max_objects"]) if form_data.get(
                "max_objects") else None,
            page=int(form_data.get("page", 1)),
            per_page=int(form_data.get("per_page", 20))
        )


class ObjectDetectionItem(BaseModel):
    score: float
    box: Tuple[float, float, float, float]

    @validator('box')
    def check_box(cls, v):
        if len(v) != 4:
            raise ValueError('box must contain exactly 4 values')
        return v


class ObjectDetection(BaseModel):
    objects: Dict[Category, List[ObjectDetectionItem]]
    counts: Dict[Category, int]


class KeyframeInfo(BaseModel):
    shot_index: int
    frame_index: int
    shot_start: int
    shot_end: int
    timestamp: float
    video_path: str
    frame_path: str

    @validator('shot_index', 'frame_index', 'shot_start', 'shot_end')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

    @validator('timestamp')
    def check_timestamp(cls, v):
        if v < 0:
            raise ValueError('Timestamp must be non-negative')
        return v


class Score(BaseModel):
    value: float = Field(0.0, ge=0.0)
    details: Dict[str, float] = Field(default_factory=dict)

    @property
    def get_value(self) -> float:
        return float(self.value)


class FrameMetadataModel(BaseModel):
    id: str
    keyframe: KeyframeInfo
    detection: Optional[ObjectDetection] = None
    score: Score = Field(default_factory=Score)
    selected: bool = Field(default=False)

    def get_corrected_frame_path(self) -> str:
        return f"keyframes/{self.keyframe.frame_path}"

    def get_corrected_video_path(self) -> str:
        return f"videos/{self.keyframe.video_path}"

    @property
    def final_score(self) -> float:
        return self.score.get_value

    @final_score.setter
    def final_score(self, value: float):
        self.score.value = value


class SearchResult(BaseModel):
    frames: List[FrameMetadataModel]
    total: int
    page: int
    has_more: bool

    @validator('total', 'page')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v
