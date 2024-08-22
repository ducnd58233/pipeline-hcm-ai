import json
from flask import current_app


class FrameMetadata:
    @staticmethod
    def get(frame_id):
        with open(current_app.config['KEYFRAMES_METADATA_PATH'], 'r') as f:
            metadata = json.load(f)

        return metadata.get(frame_id)
