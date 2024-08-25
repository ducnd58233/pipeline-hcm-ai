import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getcwd()


class Config:
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')

    KEYFRAMES_DIR = f'{BASE_DIR}/notebooks/data_extraction/transnet/Keyframes'
    VIDEOS_DIR = f'{BASE_DIR}/notebooks/data_extraction/dataset/AIC_Video'

    METADATA_PATH = f'{BASE_DIR}/notebooks/final_metadata.json'
    FAISS_BIN_PATH = f'{BASE_DIR}/notebooks/indexing/faiss_clipv2_cosine_cpu.bin'
    RESULTS_CSV_PATH = f'{BASE_DIR}/results.csv'

    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))

    CLIP_MODEL_NAME = "ViT-L-14"
    USER_ID = "default_user"
