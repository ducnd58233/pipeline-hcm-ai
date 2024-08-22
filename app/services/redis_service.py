import redis
import json
from config import Config


class RedisService:
    def __init__(self):
        self.redis_client = redis.from_url(Config.REDIS_URL)

    def save_search_history(self, user_id, search_data):
        key = f"search_history:{user_id}"
        self.redis_client.lpush(key, json.dumps(search_data))
        self.redis_client.ltrim(key, 0, 19)  # Keep only the last 20 searches

    def get_search_history(self, user_id):
        key = f"search_history:{user_id}"
        history = self.redis_client.lrange(key, 0, -1)
        return [json.loads(item) for item in history]

    def save_selected_frames(self, user_id, frame_data):
        key = f"selected_frames:{user_id}"
        self.redis_client.lpush(key, json.dumps(frame_data))

    def get_selected_frames(self, user_id):
        key = f"selected_frames:{user_id}"
        frames = self.redis_client.lrange(key, 0, -1)
        return [json.loads(item) for item in frames]
