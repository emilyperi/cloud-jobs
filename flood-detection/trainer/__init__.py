import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_ENV = os.environ.get('TRAIN_ENV', 'cloud')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
IMG_FILE_NAME = os.environ.get('IMG_FILE_NAME', 'scaled_clahe_rgb_images_filtered.npy')
LABELS_FILE_NAME = os.environ.get('LABELS_FILE_NAME', 's2_labels_filtered.npy')

# Paths for cloud runs (Vertex AI)
CLOUD_MODEL_DIR = os.environ.get('AIP_MODEL_DIR', None)
CLOUD_CHECKPOINT_DIR = os.environ.get('AIP_CHECKPOINT_DIR', None)
CLOUD_TENSORBOARD_LOG_DIR = os.environ.get('AIP_TENSORBOARD_LOG_DIR', None)
BUCKET = os.environ.get('BUCKET', 'flood-data-11-29-23')

# Default local paths
LOCAL_MODEL_DIR = f"model/{TIMESTAMP}"
LOCAL_PLOT_DIR = f"plots/{TIMESTAMP}"
LOCAL_LOG_DIR = f"logs/{TIMESTAMP}"
LOCAL_TENSORBOARD_DIR = f"tensorboard/{TIMESTAMP}"
LOCAL_CHECKPOINTS_DIR = f"checkpoints"

DATA_PATH = os.environ.get('DATA_PATH', 'data')

if TRAIN_ENV.lower() == 'local':
    JOB_ID = os.environ.get("JOB_ID", TIMESTAMP)
    MODEL_DIR = LOCAL_MODEL_DIR
    PLOT_DIR = LOCAL_PLOT_DIR
    TENSORBOARD_DIR = LOCAL_TENSORBOARD_DIR
    CHECKPOINT_DIR = LOCAL_CHECKPOINTS_DIR

    os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    IMG_FILE_PATH = f"{DATA_PATH}/{IMG_FILE_NAME}"
    LABELS_FILE_PATH = f"{DATA_PATH}/{LABELS_FILE_NAME}"


else:
    JOB_ID = os.environ.get('CLOUD_ML_JOB_ID')

    MODEL_DIR = CLOUD_MODEL_DIR
    PLOT_DIR = f"/gcs/{BUCKET}/plots"
    TENSORBOARD_DIR = CLOUD_TENSORBOARD_LOG_DIR
    CHECKPOINT_DIR = CLOUD_CHECKPOINT_DIR

    IMG_FILE_PATH = f"/gcs/{BUCKET}/{DATA_PATH}/{IMG_FILE_NAME}"
    LABELS_FILE_PATH = f"/gcs/{BUCKET}/{DATA_PATH}/{LABELS_FILE_NAME}"

CHECKPOINT_TEMPLATE = "model-{epoch:02d}-{val_loss:.2f}.ckpt"


config = {
    "TRAIN_ENV": TRAIN_ENV,
    "JOB_ID": JOB_ID,
    "MODEL_DIR": MODEL_DIR,
    "PLOT_DIR": PLOT_DIR,
    "TENSORBOARD_DIR": TENSORBOARD_DIR,
    "CHECKPOINT_DIR": f"{CHECKPOINT_DIR}/{JOB_ID}",
    "IMG_FILE_PATH": IMG_FILE_PATH,
    "LABELS_FILE_PATH": LABELS_FILE_PATH,
    "CHECKPOINT_TEMPLATE": CHECKPOINT_TEMPLATE,
    "BUCKET": BUCKET,
    "LOCAL_LOG_DIR": LOCAL_LOG_DIR,
    "LOG_LEVEL": LOG_LEVEL

}
