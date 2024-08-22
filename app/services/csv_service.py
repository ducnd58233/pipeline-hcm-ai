import csv
from config import Config
import logging
import asyncio

logger = logging.getLogger(__name__)


async def save_to_csv(selected_frames):
    logger.info(f"Saving {len(selected_frames)} frames to CSV")
    await asyncio.to_thread(_write_csv, selected_frames)
    logger.info(f"CSV saved to {Config.RESULTS_CSV_PATH}")


def _write_csv(selected_frames):
    with open(Config.RESULTS_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_id', 'frame_path'])
        for frame_id in selected_frames:
            # You might want to fetch this from your database
            frame_path = f"/path/to/{frame_id}.jpg"
            writer.writerow([frame_id, frame_path])
