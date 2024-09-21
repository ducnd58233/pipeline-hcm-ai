import csv
import os
from typing import Dict, List
from app.models import Tag
from app.log import logger
from config import Config

logger = logger.getChild(__name__)


def load_tags_from_csv(file_path: str) -> List[str]:
    logger.info(f'Loaded tags from: {file_path}')
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        data = sorted([row[0].strip() for row in reader])
        logger.info(f'Total tags: {len(data)}')
        return data


tags_list = load_tags_from_csv(os.path.join(Config.TAG_ENCODED_DIR, 'tags.csv'))

class TagManager:
    def process_tagging(self, frame_id: str, tag_data: Dict) -> Tag:
        tag_key = f"{frame_id}_tag"
        if tag_key not in tag_data:
            logger.debug(f"No tag data for {frame_id}")
            return Tag(taggers=[])
        
        tag_info = tag_data[tag_key]
        logger.debug(f"Raw tag data for {frame_id}: {tag_info}")
        
        return Tag(taggers=tag_info)