import os
from trainer.env import TIMESTAMP, TUNING, DATA_AUGMENT, MODEL_ID, CHECKPOINT_TEMPLATE

RUN = 3

# Default local paths
MODEL_DIR = f"model/run{RUN}"
PLOT_DIR = f"plots/run{RUN}"
LOG_DIR = f"logs/run{RUN}"
TENSORBOARD_DIR = f"tensorboard/run{RUN}"
CHECKPOINT_DIR = f"checkpoints"

JOB_ID = os.environ.get("JOB_ID", TIMESTAMP)

DATA_DIR = 'data/s1_data'
METADATA_DIR = f"model/run{RUN}"
LOAD_MODEL_FROM = os.environ.get("LOAD_MODEL_FROM")

RAW_IMG_FILE_PATH = os.path.join(DATA_DIR, "s1_resampled_quality_clipped_flood_images.npy")
RAW_LABELS_FILE_PATH = os.path.join(DATA_DIR, "s1_flood_quality_labels.npy")

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/train')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/validation')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'k-folds/test')

TRAIN_IMG_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_images_{MODEL_ID}.npy")
TRAIN_LABELS_FILE_PATH = os.path.join(TRAIN_DATA_DIR, f"train_labels_{MODEL_ID}.npy")
VAL_IMG_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_images_{MODEL_ID}.npy")
VAL_LABELS_FILE_PATH = os.path.join(VAL_DATA_DIR, f"val_labels_{MODEL_ID}.npy")
TEST_IMG_FILE_PATH = os.path.join(TEST_DATA_DIR, "test_images.npy")
TEST_LABELS_FILE_PATH = os.path.join(TEST_DATA_DIR, "test_labels.npy")

SAVED_MODEL_TEMPLATE = "sar_model_fold_{}.keras"

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

local_config = {
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
    "MODEL_ID": MODEL_ID,
    "METADATA_DIR": METADATA_DIR,
    "TIMESTAMP": TIMESTAMP,
    "SAVED_MODEL_TEMPLATE": SAVED_MODEL_TEMPLATE,
    "RAW_IMG_FILE_PATH": RAW_IMG_FILE_PATH,
    "RAW_LABELS_FILE_PATH": RAW_LABELS_FILE_PATH,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,
    "VAL_DATA_DIR": VAL_DATA_DIR
}
