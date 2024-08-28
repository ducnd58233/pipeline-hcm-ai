from typing import Optional, Dict, List, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum


class QueryLogic(str, Enum):
    AND = "and"
    OR = "or"


class Category(str, Enum):
    AIRPLANE = "airplane"
    APPLE = "apple"
    BACKPACK = "backpack"
    BANANA = "banana"
    BASEBALL_BAT = "baseball bat"
    BASEBALL_GLOVE = "baseball glove"
    BEAR = "bear"
    BED = "bed"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BIRD = "bird"
    BOAT = "boat"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BROCCOLI = "broccoli"
    BUS = "bus"
    CAKE = "cake"
    CAR = "car"
    CARROT = "carrot"
    CAT = "cat"
    CELL_PHONE = "cell phone"
    CHAIR = "chair"
    CLOCK = "clock"
    COUCH = "couch"
    COW = "cow"
    CUP = "cup"
    DINING_TABLE = "dining table"
    DOG = "dog"
    DONUT = "donut"
    ELEPHANT = "elephant"
    FIRE_HYDRANT = "fire hydrant"
    FORK = "fork"
    FRISBEE = "frisbee"
    HAIR_DRIER = "hair drier"
    HANDBAG = "handbag"
    HORSE = "horse"
    HOT_DOG = "hot dog"
    KEYBOARD = "keyboard"
    KITE = "kite"
    KNIFE = "knife"
    LAPTOP = "laptop"
    MICROWAVE = "microwave"
    MOTORCYCLE = "motorcycle"
    MOUSE = "mouse"
    ORANGE = "orange"
    OVEN = "oven"
    PARKING_METER = "parking meter"
    PERSON = "person"
    PIZZA = "pizza"
    POTTED_PLANT = "potted plant"
    REFRIGERATOR = "refrigerator"
    REMOTE = "remote"
    SANDWICH = "sandwich"
    SCISSORS = "scissors"
    SHEEP = "sheep"
    SINK = "sink"
    SKATEBOARD = "skateboard"
    SKIS = "skis"
    SNOWBOARD = "snowboard"
    SPOON = "spoon"
    SPORTS_BALL = "sports ball"
    STOP_SIGN = "stop sign"
    SUITCASE = "suitcase"
    SURFBOARD = "surfboard"
    TEDDY_BEAR = "teddy bear"
    TENNIS_RACKET = "tennis racket"
    TIE = "tie"
    TOILET = "toilet"
    TOOTHBRUSH = "toothbrush"
    TRAFFIC_LIGHT = "traffic light"
    TRAIN = "train"
    TRUCK = "truck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINE_GLASS = "wine glass"
    ZEBRA = "zebra"


class ObjectQuery(BaseModel):
    objects: Dict[str, Category] = Field(default_factory=dict)
    logic: QueryLogic = QueryLogic.AND
    max_objects: Optional[int] = None

    @classmethod
    def from_query_params(cls, objects: Dict[str, str], logic: str, max_objects: Optional[str] = None) -> 'ObjectQuery':
        parsed_objects = {k: Category(v) for k, v in objects.items()}
        return cls(
            objects=parsed_objects,
            logic=logic,
            max_objects=int(max_objects) if max_objects is not None else None
        )

class TextQuery(BaseModel):
    query: str


class Searcher(BaseModel):
    query: Union[TextQuery, ObjectQuery]
    weight: float = Field(..., ge=0.0, le=1.0)

    @validator('weight')
    def check_weight(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Weight must be between 0 and 1')
        return v


class QueriesStructure(BaseModel):
    text_searcher: Optional[Searcher] = None
    object_detection_searcher: Optional[Searcher] = None


class SearchRequest(BaseModel):
    queries: QueriesStructure
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)

    @classmethod
    def from_form(cls, form_data: Dict[str, str]) -> 'SearchRequest':
        text_query = form_data.get("text_query")
        text_weight = float(form_data.get("text_weight", 0.5))

        objects = {}
        for key, value in form_data.items():
            if key.startswith("objects.") and value:
                row, col = key.split(".")[1:]
                objects[f"{row},{col}"] = Category(value)

        object_weight = float(form_data.get("object_weight", 0.5))

        queries = QueriesStructure(
            text_searcher=Searcher(query=TextQuery(
                query=text_query), weight=text_weight) if text_query else None,
            object_detection_searcher=Searcher(query=ObjectQuery(
                objects=objects), weight=object_weight) if objects else None
        )

        return cls(
            queries=queries,
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
