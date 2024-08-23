import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getcwd()


class Config:
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')

    KEYFRAMES_DIR = f'{BASE_DIR}/notebooks/data_extraction/transnet/Keyframes'
    VIDEOS_DIR = f'{BASE_DIR}/notebooks/data_extraction/dataset/AIC_Video'
    
    KEYFRAMES_METADATA_PATH = f'{BASE_DIR}/notebooks/keyframes_metadata.json'
    FAISS_BIN_PATH = f'{BASE_DIR}/notebooks/indexing/faiss_clipv2_cosine_cpu.bin'
    RESULTS_CSV_PATH = f'{BASE_DIR}/results.csv'

    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
