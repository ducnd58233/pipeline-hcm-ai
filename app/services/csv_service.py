import os
import csv
from app.models import FrameMetadataModel
from config import Config
import logging
import asyncio

logger = logging.getLogger(__name__)


async def save_single_frame_to_csv(frame: FrameMetadataModel):
    logger.info(f"Saving frame {frame.id} to CSV")
    await asyncio.to_thread(_write_single_frame_csv, frame)
    logger.info(f"CSV saved to {Config.RESULTS_CSV_PATH}")


def _write_single_frame_csv(frame: FrameMetadataModel):
    file_exists = os.path.isfile(Config.RESULTS_CSV_PATH)
    with open(Config.RESULTS_CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['frame_id', 'frame_path', 'score'])
        writer.writerow(
            [frame.id, f"keyframes/{frame.keyframe.frame_path}", frame.score])
