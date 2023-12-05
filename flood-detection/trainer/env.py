import os
import tempfile
from datetime import datetime


TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_ENV = os.environ.get('TRAIN_ENV', 'cloud')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

K_FOLD = os.environ.get('K_FOLD', '0').lower() == '1'
GPU_ENABLED = os.environ.get('GPU_ENABLED', '0').lower() == '1'


# Paths for cloud runs (Vertex AI)
CLOUD_MODEL_DIR = os.environ.get('AIP_MODEL_DIR', None)
CLOUD_CHECKPOINT_DIR = os.environ.get('AIP_CHECKPOINT_DIR', None)
CLOUD_TENSORBOARD_LOG_DIR = os.environ.get('AIP_TENSORBOARD_LOG_DIR', None)
BUCKET = os.environ.get('BUCKET', 'flood-data-11-29-23')

# Default local paths
LOCAL_MODEL_DIR = "model"
LOCAL_PLOT_DIR = "plots"
LOCAL_LOG_DIR = "logs"
LOCAL_TENSORBOARD_DIR = f"tensorboard"
LOCAL_CHECKPOINTS_DIR = f"checkpoints"

DATA_DIR = os.environ.get('DATA_PATH', 'data')
MODEL_ID = os.environ.get("MODEL_ID", 0)


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

    TEMP_LOG_FILE_NAME = ''
else:
    JOB_ID = os.environ.get('CLOUD_ML_JOB_ID')

    MODEL_DIR = CLOUD_MODEL_DIR
    PLOT_DIR = f"/gcs/{BUCKET}/plots"
    TENSORBOARD_DIR = CLOUD_TENSORBOARD_LOG_DIR
    CHECKPOINT_DIR = CLOUD_CHECKPOINT_DIR
    DATA_DIR = f"/gcs/{BUCKET}/data"

    TEMP_LOG_FILE_NAME = tempfile.NamedTemporaryFile(mode='w', delete=False).name


if K_FOLD:
    IMG_FILE_NAME = os.environ.get('IMG_FILE_NAME', 'scaled_clahe_rgb_images_filtered.npy')
    LABELS_FILE_NAME = os.environ.get('LABELS_FILE_NAME', 's2_labels_filtered.npy')
    TRAIN_IMG_FILE_PATH = os.path.join(DATA_DIR, IMG_FILE_NAME)
    TRAIN_LABELS_FILE_PATH = os.path.join(DATA_DIR, LABELS_FILE_NAME)
    VAL_IMG_FILE_PATH = None
    VAL_LABELS_FILE_PATH = None
else:
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/train')
    VAL_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/validation')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/test')
    TRAIN_IMG_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_images_{MODEL_ID}.npy")
    TRAIN_LABELS_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_labels_{MODEL_ID}.npy")
    VAL_IMG_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_images_{MODEL_ID}.npy")
    VAL_LABELS_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_labels_{MODEL_ID}.npy")

CHECKPOINT_TEMPLATE = "model-{epoch:02d}-{val_loss:.2f}.ckpt"


config = {
    "TRAIN_ENV": TRAIN_ENV,
    "JOB_ID": JOB_ID,
    "MODEL_DIR": MODEL_DIR,
    "PLOT_DIR": PLOT_DIR,
    "TENSORBOARD_DIR": TENSORBOARD_DIR,
    "CHECKPOINT_DIR": os.path.join(CHECKPOINT_DIR, JOB_ID),
    "TRAIN_IMG_FILE_PATH": TRAIN_IMG_FILE_PATH,
    "TRAIN_LABELS_FILE_PATH": TRAIN_LABELS_FILE_PATH,
    "VAL_IMG_FILE_PATH": VAL_IMG_FILE_PATH,
    "VAL_LABELS_FILE_PATH": VAL_LABELS_FILE_PATH,
    "CHECKPOINT_TEMPLATE": CHECKPOINT_TEMPLATE,
    "BUCKET": BUCKET,
    "LOCAL_LOG_DIR": LOCAL_LOG_DIR,
    "LOG_LEVEL": LOG_LEVEL,
    "TEMP_LOG_FILE": TEMP_LOG_FILE_NAME,
    "TEMP_META_DATA_FILES": [],
    "MODEL_ID": MODEL_ID,
    "K_FOLD": K_FOLD,
    "METADATA_DIR": LOCAL_MODEL_DIR,
    "TIMESTAMP": TIMESTAMP

}
