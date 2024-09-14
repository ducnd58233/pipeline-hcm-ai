import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getcwd()


class Config:
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')

    KEYFRAMES_DIR = f'{BASE_DIR}/notebooks/data_extraction/transnet/keyframes'
    VIDEOS_DIR = f'{BASE_DIR}/notebooks/data_extraction/dataset/AIC_Video'
    METADATA_ENCODED_DIR = f'{BASE_DIR}/notebooks/indexing/metadata_encoded'
    
    OD_ENCODED_DIR = f'{METADATA_ENCODED_DIR}/object_detection'
    TAG_ENCODED_DIR = f'{METADATA_ENCODED_DIR}/multi_tag'

    METADATA_DIR = f'{BASE_DIR}/notebooks'
    FAISS_BIN_PATH = f'{BASE_DIR}/notebooks/indexing/faiss_clipv2_cosine_cpu.bin'
    RESULTS_DIR = f'{BASE_DIR}'

    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))

    CLIP_MODEL_NAME = "ViT-L-14"
    USER_ID = "default_user"
    
    MAX_FRAMES_PER_FILE = 100
