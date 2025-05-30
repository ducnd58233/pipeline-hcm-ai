import os
import csv
from typing import List, Dict
from app.models import FrameMetadataModel
from app.utils.data_manager.frame_data_manager import frame_data_manager
from config import Config
from app.log import logger
import asyncio

logger = logger.getChild(__name__)


async def save_single_frame_to_csv(frame: FrameMetadataModel, file_name: str):
    logger.info(f"Saving frame {frame.id} to CSV file: {file_name}")
    await asyncio.to_thread(add_frame_to_file, frame, file_name)
    logger.info(f"CSV saved to {os.path.join(Config.RESULTS_DIR, file_name)}")


def add_frame_to_file(frame: FrameMetadataModel, file_name: str):
    file_path = os.path.join(Config.RESULTS_DIR, file_name)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(
        #     [frame.id, f"keyframes/{frame.keyframe.frame_path}", frame.final_score])
        video_id, frame_idx = frame.get_frame_info()
        writer.writerow([video_id, frame_idx])


def remove_frame_from_file(frame_id: str, file_name: str):
    file_path = os.path.join(Config.RESULTS_DIR, file_name)
    temp_file_path = os.path.join(Config.RESULTS_DIR, f"temp_{file_name}")
    frame_id_part = frame_id.split('_')
    video_id = '_'.join(frame_id_part[:2])
    frame_idx = int(frame_id_part[-1])

    with open(file_path, 'r') as csvfile, open(temp_file_path, 'w', newline='') as temp_file:
        reader = csv.reader(csvfile)
        writer = csv.writer(temp_file)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if row[0] != video_id and int(row[1]) != frame_idx:
                writer.writerow(row)

    os.replace(temp_file_path, file_path)


def get_existing_csv_files() -> List[str]:
    return [f for f in os.listdir(Config.RESULTS_DIR) if f.endswith('.csv')]


def create_new_csv_file(file_name: str) -> str:
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    file_path = os.path.join(Config.RESULTS_DIR, file_name)
    counter = 1
    while os.path.exists(file_path):
        file_name = f"{file_name[:-4]}_{counter}.csv"
        file_path = os.path.join(Config.RESULTS_DIR, file_name)
        counter += 1

    with open(file_path, 'w', newline='') as csvfile:
        csv.writer(csvfile)
        # writer.writerow(['frame_id', 'frame_path', 'score'])

    return file_name


def is_frame_in_csv(frame_id: str, file_name: str) -> bool:
    file_path = os.path.join(Config.RESULTS_DIR, file_name)
    if not os.path.exists(file_path):
        return False

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)  # Skip header
        # return any(row[0] == frame_id for row in reader)
        frame_part = frame_id.split('_')
        video_id = '_'.join(frame_part[:2])
        frame_idx = int(frame_part[-1])
        return any(row[0] == video_id and int(row[1]) == frame_idx for row in reader)


def get_file_contents(file_name: str) -> Dict:
    file_path = os.path.join(Config.RESULTS_DIR, file_name)
    existing_frames = []
    frame_ids = set()

    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # next(reader)  # Skip header
            for row in reader:
                frame_id = f'{row[0]}_{int(row[1]):06d}'
                frame_id_extra = f'{row[0]}_extra_{int(row[1]):06d}'
                frame = frame_data_manager.get_frame_by_key(
                    frame_id) or frame_data_manager.get_frame_by_key(frame_id_extra)
                logger.info(f'frame_id: {frame_id} - frame_id_extra: {frame_id_extra} - frame: {frame}')
                if frame:
                    existing_frames.append(frame)
                    frame_ids.add(frame_id)

    frames_to_add = [frame for frame in frame_data_manager.get_selected_frames(
    ) if frame.id not in frame_ids]
    limit_exceeded = len(existing_frames) + \
        len(frames_to_add) > Config.MAX_FRAMES_PER_FILE

    return {
        "existing_frames": existing_frames,
        "frames_to_add": frames_to_add,
        "limit_exceeded": limit_exceeded
    }
