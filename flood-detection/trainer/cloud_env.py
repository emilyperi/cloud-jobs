import os
from trainer.env import TIMESTAMP, TUNING, DATA_AUGMENT, MODEL_ID, CHECKPOINT_TEMPLATE, SAVED_MODEL_TEMPLATE

# Paths for cloud runs (Vertex AI)
CLOUD_MODEL_DIR = os.environ.get('AIP_MODEL_DIR', "model")
CLOUD_CHECKPOINT_DIR = os.environ.get('AIP_CHECKPOINT_DIR', "checkpoints")
CLOUD_TENSORBOARD_LOG_DIR = os.environ.get('AIP_TENSORBOARD_LOG_DIR', "tensorboard")
BUCKET = os.environ.get('BUCKET', 'sar-flood-detection')

LOAD_MODEL_FROM = os.environ.get("LOAD_MODEL_FROM")

JOB_ID = os.environ.get('CLOUD_ML_JOB_ID', TIMESTAMP)

MODEL_DIR = CLOUD_MODEL_DIR
PLOT_DIR = f"/gcs/{BUCKET}/plots"
TENSORBOARD_DIR = CLOUD_TENSORBOARD_LOG_DIR
CHECKPOINT_DIR = CLOUD_CHECKPOINT_DIR
DATA_DIR = f"/gcs/{BUCKET}/data/k-folds"
METADATA_DIR = f"/gcs/{BUCKET}/model"


TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_IMG_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_images_{MODEL_ID}.npy")
TRAIN_LABELS_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_labels_{MODEL_ID}.npy")
VAL_IMG_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_images_{MODEL_ID}.npy")
VAL_LABELS_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_labels_{MODEL_ID}.npy")
TEST_IMG_FILE_PATH = os.path.join(TEST_DATA_DIR, "test_images.npy")
TEST_LABELS_FILE_PATH = os.path.join(TEST_DATA_DIR, "test_labels.npy")


if LOAD_MODEL_FROM:
    MODEL_LOAD_DIR = os.path.join(MODEL_DIR, LOAD_MODEL_FROM)
else:
    MODEL_LOAD_DIR = MODEL_DIR

MODEL_SAVE_DIR = MODEL_DIR

if TUNING:
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "tuned")
    TENSORBOARD_DIR = os.path.join(TENSORBOARD_DIR, "tuned")
    MODEL_SAVE_DIR = os.path.join(MODEL_DIR, "tuned")
    METADATA_DIR = os.path.join(METADATA_DIR, "tuned")
    PLOT_DIR = os.path.join(PLOT_DIR, "tuned")

if DATA_AUGMENT:
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "data_augment")
    TENSORBOARD_DIR = os.path.join(TENSORBOARD_DIR, "data_augment")
    METADATA_DIR = os.path.join(METADATA_DIR, "data_augment")
    MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "data_augment")
    PLOT_DIR = os.path.join(PLOT_DIR, "data_augment")

cloud_config = {
    "JOB_ID": JOB_ID,
    "MODEL_SAVE_DIR": MODEL_SAVE_DIR,
    "MODEL_LOAD_DIR": MODEL_LOAD_DIR,
    "PLOT_DIR": PLOT_DIR,
    "TENSORBOARD_DIR": TENSORBOARD_DIR,
    "CHECKPOINT_DIR": os.path.join(CHECKPOINT_DIR, JOB_ID),
    "TRAIN_IMG_FILE_PATH": TRAIN_IMG_FILE_PATH,
    "TRAIN_LABELS_FILE_PATH": TRAIN_LABELS_FILE_PATH,
    "VAL_IMG_FILE_PATH": VAL_IMG_FILE_PATH,
    "VAL_LABELS_FILE_PATH": VAL_LABELS_FILE_PATH,
    "TEST_IMG_FILE_PATH": TEST_IMG_FILE_PATH,
    "TEST_LABELS_FILE_PATH": TEST_LABELS_FILE_PATH,
    "CHECKPOINT_TEMPLATE": CHECKPOINT_TEMPLATE,
    "BUCKET": BUCKET,
    "TEMP_META_DATA_FILES": [],
    "MODEL_ID": MODEL_ID,
    "METADATA_DIR": METADATA_DIR,
    "TIMESTAMP": TIMESTAMP,
    "SAVED_MODEL_TEMPLATE": SAVED_MODEL_TEMPLATE
}
