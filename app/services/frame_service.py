from app.services.redis_service import redis_service
from typing import Set


class FrameService:
    def __init__(self):
        self.redis_service = redis_service

    async def toggle_frame_selection(self, user_id: str, frame_id: str) -> bool:
        selected_frames = await self.__get_selected_frames(user_id)
        key = f"selected_frames:{user_id}"
        if frame_id in selected_frames:
            await self.redis_service.remove_from_set(key, frame_id)
            return False
        else:
            await self.redis_service.add_to_set(key, frame_id)
            return True

    async def clear_all_selected_frames(self, user_id: str):
        await self.redis_service.clear_set(f"selected_frames:{user_id}")

    async def retrieve_selected_frames(self, user_id: str) -> Set[int]:
        selected_frames = await self.__get_selected_frames(user_id)
        return selected_frames

    async def __get_selected_frames(self, user_id: str):
        selected_frames = {frame_id for frame_id in await self.redis_service.get_set_members(
            f"selected_frames:{user_id}")}
        return selected_frames


frame_service = FrameService()
