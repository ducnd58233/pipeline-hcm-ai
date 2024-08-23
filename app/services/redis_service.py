import redis
from typing import Set


class RedisService:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db)

    async def add_to_set(self, key: str, value: str):
        self.client.sadd(key, value)

    async def remove_from_set(self, key: str, value: str):
        self.client.srem(key, value)

    async def get_set_members(self, key: str) -> Set[str]:
        return self.client.smembers(key)

    async def is_member_of_set(self, key: str, value: str) -> bool:
        return self.client.sismember(key, value)

    async def clear_set(self, key: str):
        self.client.delete(key)

redis_service = RedisService()