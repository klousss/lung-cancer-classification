import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

RANDOM_STATE = 42
TEST_SIZE = 0.2
