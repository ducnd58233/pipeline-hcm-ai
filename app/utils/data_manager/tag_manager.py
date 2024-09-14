from typing import Dict, List
from app.models import Tag
from app.log import logger

logger = logger.getChild(__name__)

class TagManager:
    def process_tagging(self, frame_id: str, tag_data: Dict) -> Tag:
        tag_key = f"{frame_id}_tag"
        if tag_key not in tag_data:
            logger.debug(f"No tag data for {frame_id}")
            return Tag(taggers=[])
        
        tag_info = tag_data[tag_key]
        logger.debug(f"Raw tag data for {frame_id}: {tag_info}")
        
        return Tag(taggers=tag_info)