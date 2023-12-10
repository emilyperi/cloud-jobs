import logging
import tempfile
import os
from trainer.env import TRAIN_ENV, TIMESTAMP, LOG_LEVEL

LOG_DIR = os.environ.get("LOG_DIR", "logs")
TEMP_LOG_FILE_NAME = tempfile.NamedTemporaryFile(mode='w', delete=False).name
LOG_SAVE_PATH = f'{LOG_DIR}/trainer{TIMESTAMP}.log'

if TRAIN_ENV.lower() == 'cloud':
    log_file_path = TEMP_LOG_FILE_NAME

else:
    log_file_path = LOG_SAVE_PATH

# Set the root logger to a higher level, e.g., WARNING
logging.basicConfig(level=logging.WARNING)

# Configure application's logger for a lower level
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# File handler - logs to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Stream handler - logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Adding handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


