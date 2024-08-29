from typing import Dict, List, Tuple
from app.models import Category, ObjectDetection, ObjectDetectionItem
from .visual_encoding_manager import VisualEncodingManager
from app.log import logger


class ObjectDetectionManager:
    def __init__(self, visual_encoding_manager: VisualEncodingManager):
        self.visual_encoding_manager = visual_encoding_manager

    def process_object_detection(self, frame_id: str, object_detection_data: Dict, frame_dimensions: Tuple[int, int]) -> ObjectDetection:
        od_key = f"{frame_id}_detection"
        if od_key not in object_detection_data:
            logger.debug(f"No object detection data for {frame_id}")
            return ObjectDetection(objects={}, counts={})

        od_info = object_detection_data[od_key]
        logger.debug(f"Raw object detection data for {frame_id}: {od_info}")

        objects = {}
        counts = {}
        encoded_detection = []
        if 'objects' in od_info and od_info['objects']:
            frame_width, frame_height = frame_dimensions

            for label, metadata in od_info['objects'].items():
                category = self._get_category(label)
                bbox_detection = self.visual_encoding_manager.encode_bboxes(
                    category, metadata, frame_width, frame_height)
                objects[category] = bbox_detection
                encoded_detection.extend(
                    [item.encoded_bbox for item in bbox_detection])

            counts = {Category(k): v for k, v in od_info['counts'].items()
                      if self._get_category(k) in Category.__members__}

        logger.debug(f'objects: {objects}, counts: {counts}')
        return ObjectDetection(objects=objects, counts=counts, encoded_detection=' '.join(encoded_detection))

    def _get_category(self, category_name: str) -> Category:
        return Category(category_name.lower())
