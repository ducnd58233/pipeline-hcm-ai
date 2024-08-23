from app.error import FrameNotFoundError
from app.services.redis_service import redis_service
from app.models import FrameMetadata, FrameMetadataModel
from typing import Any, Dict, List, Optional, Set
import json
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


class FrameService:
    def __init__(self):
        self.redis_service = redis_service
        self.timezone = pytz.timezone('Asia/Ho_Chi_Minh')

    async def toggle_frame_selection(self, user_id: str, frame_id: str, score: float) -> bool:
        try:
            key = self.__get_selected_frames_key(user_id)
            frame_key = self.__get_frame_key(frame_id)

            current_time = datetime.now(self.timezone)
            logger.info(
                f"Toggle frame selection - User: {user_id}, Frame: {frame_id}, Time: {current_time}, Timezone: {self.timezone}")

            frame = self.redis_service.get_hash(frame_key)

            if not frame:
                frame = await FrameMetadata.get_by_frame_id(frame_id)
                if not frame:
                    raise FrameNotFoundError(f"Frame not found: {frame_id}")

                frame_dict = frame.model_dump()
                frame_dict["selected"] = "true"
                frame_dict["score"] = str(score)
                self.redis_service.add_to_set(key, frame_id)
                self.redis_service.set_hash(frame_key, frame_dict)
                logger.info(f"New frame selected: {frame_id}")
                return True

            frame['selected'] = str(frame.get(
                'selected', 'true').lower() != 'true')
            frame['score'] = str(score)
            self.set_frame(frame)
            logger.info(
                f"Frame selection toggled: {frame_id}, New state: {frame['selected']}")
            return frame['selected'].lower() == 'true'

        except FrameNotFoundError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Error in toggle_frame_selection: {str(e)}")
            raise

    async def get_frame_data(self, frame_id) -> FrameMetadataModel:
        frame_key = self.__get_frame_key(frame_id)
        frame = self.redis_service.get_hash(frame_key)

        if not frame:
            frame = await FrameMetadata.get_by_frame_id(frame_id)
            self.redis_service.set_hash(frame_key, json.dumps(frame))
        else:
            frame = self.__convert_frame_data(frame)

        return FrameMetadataModel(**frame)

    def get_selected_frames_list(self, user_id: str) -> List[Optional[FrameMetadataModel]]:
        key = self.__get_selected_frames_key(user_id)
        selected_frame_ids = {
            id for id in self.redis_service.get_set_members(key)}
        return [FrameMetadataModel(**self.__convert_frame_data(self.redis_service.get_hash(self.__get_frame_key(frame_id))))
                for frame_id in selected_frame_ids]

    def set_frame(self, frame_dict: Dict[str, Any]) -> None:
        key = self.__get_frame_key(frame_dict["id"])
        string_frame_dict = {
            k: str(v) if v is not None else None for k, v in frame_dict.items()}
        self.redis_service.set_hash(key, string_frame_dict)

    def __convert_frame_data(self, frame: Dict[str, str]) -> Dict[str, Any]:
        converted_frame = frame.copy()
        if 'score' in converted_frame:
            converted_frame['score'] = float(
                converted_frame['score']) if converted_frame['score'] not in ['None', ''] else None
        if 'selected' in converted_frame:
            converted_frame['selected'] = converted_frame['selected'].lower(
            ) == 'true'
        if 'shot_index' in converted_frame:
            converted_frame['shot_index'] = int(converted_frame['shot_index'])
        if 'frame_index' in converted_frame:
            converted_frame['frame_index'] = int(
                converted_frame['frame_index'])
        if 'timestamp' in converted_frame:
            converted_frame['timestamp'] = float(converted_frame['timestamp'])
        return converted_frame

    def __get_frame_key(self, frame_id):
        return f"frame_data:{frame_id}"

    def __get_selected_frames_key(self, user_id):
        return f"selected_frames:{user_id}"

    def set_timezone(self, timezone_str: str):
        """Set the timezone for logging."""
        try:
            self.timezone = pytz.timezone(timezone_str)
            logger.info(f"Timezone set to: {self.timezone}")
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Unknown timezone: {timezone_str}")
            raise ValueError(f"Unknown timezone: {timezone_str}")


frame_service = FrameService()
