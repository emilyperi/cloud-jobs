#### To Run:

```shell
$ python3.10 -m venv venv
```

```shell
$ source venv/bin/activate
```
```shell
$ python setup.py sdist --formats=gztar
```
If installing to develop:

```shell
$ pip install -e .
```
Otherwise,
```shell
$ pip install trainer-0.1.tar.gz
```

```shell
$ python -m trainer.task --help
```

Usage:
```shell
usage: task.py [-h] [--k K] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--model-config MODEL_CONFIG] [--parameter-config PARAMETER_CONFIG] [--make-predictions] [--skip-train] [--skip-save]

This script loads data and trains a ResNet50 flood detection model.

options:
  -h, --help            Show this help message and exit
  --k K                 Number of folds for k fold validation
  --batch-size BATCH_SIZE
                        Batch size for training
  --epochs EPOCHS       Number epochs for training models
  --model-config MODEL_CONFIG
                        Path to JSON model config file
  --parameter-config PARAMETER_CONFIG
                        Path to JSON parameter config file
  --make-predictions    Make model predictions
  --skip-train          Prevent model from training
  --skip-save           Prevent model saving
```

#### Environment Variables
`TRAIN_ENV`: 'cloud' or 'local' (Defaults to 'cloud')

`MODEL_DIR`: Directory to save models, defaults to `./model`

`JOB_ID`: Cloud ml job ID (Local defaults to timestamp) 

`PLOT_DIR`': Directory to store plots, defaults to './plots'

`TENSORBOARD_DIR`: Directory to store tensorboard logs, defaults to './tensorboard'

`CHECKPOINT_DIR`: Directory to store checkpoints, defaults to './checkpoints'

`DATA_DIR`: Directory containing data to load, defaults to './data'

`IMG_FILE_NAME`: Numpy file with image data, e.g.'scaled_clahe_rgb_images_filtered.npy',

`LABELS_FILE_NAME`: Numpy file with labels data, e.g. 's2_labels_filtered.npy'

`BUCKET`: google cloud storage bucket name, e.g. `flood-data-11-29-23`,

`LOG_LEVEL`: Defaults to "INFO"
