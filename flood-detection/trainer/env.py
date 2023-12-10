import os
from datetime import datetime

TRAIN_ENV = os.environ.get('TRAIN_ENV', 'cloud')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'SINGLE').lower()
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
TUNING = os.environ.get("TUNING", "0").lower() == "1"
GPU_ENABLED = os.environ.get("GPU_ENABLED", "0").lower() == "1"
DATA_AUGMENT = os.environ.get("DATA_AUGMENT", "0") == "1"
MODEL_ID = os.environ.get("MODEL_ID", 0)
CHECKPOINT_TEMPLATE = "model-{epoch:02d}-{val_loss:.2f}.ckpt"
SAVED_MODEL_TEMPLATE = "sar_model_fold_{}"
