import redis
from typing import Set, Dict, Any


class RedisService:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def add_to_set(self, key: str, value: Any):
        """Add a value to a Redis set."""
        self.client.sadd(key, value)

    def remove_from_set(self, key: str, value: Any):
        """Remove a value from a Redis set."""
        self.client.srem(key, value)

    def get_set_members(self, key: str) -> Set[Any]:
        """Retrieve all members of a Redis set."""
        return {self._decode_value(member) for member in self.client.smembers(key)}

    def delete_key(self, key: str):
        """Delete a key from Redis."""
        self.client.delete(key)

    def set_hash(self, key: str, mapping: Dict[str, Any]):
        """Set multiple fields in a Redis hash."""
        string_mapping = {k: str(v) for k, v in mapping.items()}
        self.client.hset(key, mapping=string_mapping)

    def get_hash(self, key: str) -> Dict[str, Any]:
        """Retrieve all fields and values of a Redis hash."""
        hash_data = self.client.hgetall(key)
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in hash_data.items()}

    def delete_hash(self, key: str):
        """Delete a hash key from Redis."""
        self.client.delete(key)

    def is_member_of_set(self, key: str, value: Any) -> bool:
        """Check if a value is a member of a Redis set."""
        return self.client.sismember(key, value)

    def _decode_value(self, value: Any) -> Any:
        """Decode value from bytes to string, if possible."""
        try:
            return value.decode('utf-8')
        except AttributeError:
            return value

redis_service = RedisService()