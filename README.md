# Flood Detection from Satellite Imagery

A machine learning pipeline for detecting flooding in satellite imagery using a ResNet50 model, built as a final project for Stanford CS229 (Machine Learning).

## Overview

This project trains a binary image classifier to identify flooded regions in multispectral satellite imagery. The pipeline handles raw 16-bit satellite data, preprocesses and scales it for model input, trains a ResNet50 model with k-fold cross-validation, and supports both local and Google Cloud ML job execution.

## Features

- **ResNet50-based classifier** trained on labeled satellite imagery
- **16-bit image preprocessing** with multiple scaling strategies (basic, log, CLAHE via OpenCV)
- **K-fold cross-validation** for robust model evaluation
- **Google Cloud ML integration** for distributed training jobs
- **TensorBoard logging** for training monitoring
- **Dockerized** for reproducible environments
- **Geospatial data analysis** using GeoPandas and Folium for visualizing image coverage

## Project Structure

```
cloud-jobs/
├── flood-detection/         # Core model training module
├── analyze_data.py          # Data preprocessing and visualization utilities
├── results.py               # Results loading and evaluation
├── results_analysis.ipynb   # Results analysis notebook
├── data_info.ipynb          # Dataset exploration notebook
├── Dockerfile               # Container definition
├── config.yaml              # Training configuration (CPU)
└── config-gpu.yaml          # Training configuration (GPU)
```

## Setup

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Or install from built package
python setup.py sdist --formats=gztar
pip install trainer-0.1.tar.gz
```

## Usage

```bash
python -m trainer.task --help
```

```
options:
  --k K                         Number of folds for k-fold cross-validation
  --batch-size BATCH_SIZE       Batch size for training
  --epochs EPOCHS               Number of epochs
  --model-config MODEL_CONFIG   Path to JSON model config file
  --parameter-config            Path to JSON parameter config file
  --make-predictions            Run model predictions
  --skip-train                  Skip training (load existing model)
  --skip-save                   Skip saving model outputs
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TRAIN_ENV` | `'cloud'` or `'local'` | `'cloud'` |
| `MODEL_DIR` | Directory to save models | `./model` |
| `JOB_ID` | Cloud ML job ID | timestamp (local) |
| `PLOT_DIR` | Directory for plots | `./plots` |
| `TENSORBOARD_DIR` | TensorBoard log directory | `./tensorboard` |
| `CHECKPOINT_DIR` | Checkpoint directory | `./checkpoints` |
| `DATA_DIR` | Directory containing input data | `./data` |
| `IMG_FILE_NAME` | Numpy file with image data | — |
| `LABELS_FILE_NAME` | Numpy file with labels | — |
| `BUCKET` | Google Cloud Storage bucket name | — |
| `LOG_LEVEL` | Logging level | `INFO` |

## Tech Stack

- Python, TensorFlow, NumPy
- GeoPandas, Folium, OpenCV, scikit-image
- Google Cloud ML
- Docker
