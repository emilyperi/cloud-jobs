import logging
from google.cloud import storage
from trainer.env import config

if config["TRAIN_ENV"].lower() == 'cloud':
    log_file_path = config["TEMP_LOG_FILE"]
else:
    log_file_path = f'{config["LOCAL_LOG_DIR"]}/trainer{config["TIMESTAMP"]}.log'

# Set the root logger to a higher level, e.g., WARNING
logging.basicConfig(level=logging.WARNING)

# Configure application's logger for a lower level
logger = logging.getLogger(__name__)
logger.setLevel(config["LOG_LEVEL"])

# File handler - logs to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Stream handler - logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Adding handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f"Job configuration values: {config}")


def upload_blob(bucket_name=config["BUCKET"], source_file_name=log_file_path, destination_blob_name=f'logs/trainer{config["TIMESTAMP"]}.log'):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
