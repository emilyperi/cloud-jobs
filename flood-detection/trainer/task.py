import argparse
import os.path
from pathlib import Path
import tempfile

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from trainer.env import config
from trainer.dtypes import Parameters, ModelConfig, DataSource, DataType
from trainer.model import KModels, Model
from trainer.logging import logger, upload_blob
from trainer.utils import split_indices, save_models, format_filename, save_meta_data, get_k_predictions, \
    plot_precision_recall, load_json_config, k_score_summary


def get_args():
    description = 'This script loads data and trains a ResNet50 flood detection model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='Number of folds for k fold validation')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number epochs for training models'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        required=False,
        help='Path to JSON model config file'
    )
    parser.add_argument(
        '--parameter-config',
        type=str,
        required=False,
        help='Path to JSON parameter config file'
    )
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save a PR Curve plot of the validation predictions'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skips training the model'
    )
    parser.add_argument(
        '--skip-save',
        action='store_true',
        help='Skips saving the model'
    )
    parser.add_argument(
        '--make-predictions',
        action='store_true',
        help='Makes predictions on the loaded model(s)'
    )

    return parser.parse_args()


def get_config_file_path(config_name):
    if config_name:
        package_dir = Path(__file__).parent  # Path to the directory containing this file
        config_path = package_dir / 'configs' / config_name
        return config_path


def is_cloud_env():
    return config["TRAIN_ENV"].lower() == "cloud"


def single_model(params: Parameters, model_config: ModelConfig, skip_train):
    try:
        logger.info("Loading data")
        train_labels = np.load(config["TRAIN_LABELS_FILE_PATH"])
        train_images = np.load(config["TRAIN_IMG_FILE_PATH"])
        val_labels = np.load(config["VAL_LABELS_FILE_PATH"])
        val_images = np.load(config["VAL_IMG_FILE_PATH"])
    except IOError:
        logger.exception("Failed to load data")
        return

    data = dict(train_images=train_images,
                train_labels=train_labels,
                val_images=val_images,
                val_labels=val_labels)

    model = Model(model_id=config["MODEL_ID"])
    model.model = model_config

    if not skip_train:
        try:
            logger.info("Beginning Training")
            model.train(train_data=(train_images, train_labels), val_data=(val_images, val_labels), params=params)
            metric_key = params.metric_key()
            score = model.get_score(metric_key)
            logger.info(f'Model Summary:\n'
                        f'Train Loss: {score.train_loss}\n'
                        f'Validation Loss: {score.val_loss}\n'
                        f'Train Metric ({metric_key}): {score.train_metric}\n'
                        f'Validation Metric ({metric_key}): {score.val_metric}')
        except Exception:
            logger.exception("Training failed")
            return
    return model, data


def k_models(params: Parameters, model_config: ModelConfig, skip_train):
    try:
        logger.info("Loading data")
        labels = np.load(config["TRAIN_LABELS_FILE_PATH"])
        images = np.load(config["TRAIN_IMG_FILE_PATH"])
    except IOError:
        logger.exception("Failed to load data")
        return

    try:
        logger.info("Preparing images and labels")
        preprocessed_images = preprocess_input(images)
        indices = np.arange(len(images))
        train_indices, test_indices = split_indices(indices, labels)
        train_images, test_images = preprocessed_images[train_indices], preprocessed_images[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        data = dict(train_images=train_images,
                    train_labels=train_labels,
                    test_images=test_images,
                    test_labels=test_labels)

    except Exception:
        logger.exception("Failed to preprocess or split data into train and test sets")
        return

    k_fold_model = KModels(k=params.k)
    k_fold_model.init_models(model_config)

    if not skip_train:
        try:
            logger.info("Beginning Training")
            k_fold_model.kfold_train(train_data=(train_images, train_labels), params=params)
            logger.info(k_score_summary(scores=k_fold_model.scores, metric_name=params.metric_key()))
        except Exception:
            logger.exception("Training failed")
            return
    return k_fold_model, data


def main():
    args = get_args()
    k = args.k
    batch_size = args.batch_size
    epochs = args.epochs
    model_config_file = get_config_file_path(args.model_config)
    parameter_config_file = get_config_file_path(args.parameter_config)
    save_plot = args.save_plot
    skip_train = args.skip_train
    skip_save = args.skip_save

    parameters = None
    model_config = None

    try:
        logger.info("Initializing model configuration")
        if model_config_file:
            model_config = ModelConfig.deserialize(config=load_json_config(model_config_file))
        else:
            model_config = ModelConfig(layers=[Flatten(), Dense(256, activation='relu'), Dense(1)],
                                       input_shape=(512, 512, 3))

        logger.info("Initializing model parameters")
        if parameter_config_file:
            parameters = Parameters.deserialize(config=load_json_config(parameter_config_file))
        else:
            parameters = Parameters(batch_size=batch_size,
                                    epochs=epochs,
                                    optimizer=Adam(),
                                    loss=BinaryCrossentropy(from_logits=True),
                                    metrics=[BinaryAccuracy()],
                                    k=k)
    except Exception:
        logger.exception(f"Failed to initialize model params: {parameters}, model configuration: {model_config}")
        return

    if config["K_FOLD"]:
        model, data = k_models(parameters, model_config, skip_train)
    else:
        model, data = single_model(parameters, model_config, skip_train)
    if not model:
        return

    if isinstance(model, KModels):
        models_list = model.models
    else:
        models_list = [model]

    if not skip_save:
        try:
            save_models(models=models_list, data_source=DataSource.RBG, base_path=config["MODEL_DIR"],
                        env=config["TRAIN_ENV"])
        except Exception as e:
            logger.exception(f"Saving model failed: {str(e)}")

        try:
            logger.info("Saving Model Meta Data")

            for model in models_list:
                if is_cloud_env():
                    save_path = tempfile.NamedTemporaryFile(mode='w', delete=False).name
                    config["TEMP_META_DATA_FILES"].append(save_path)
                else:
                    suffix = f"fold_{model.model_id}"
                    file_name = format_filename(DataType.METADATA, DataSource.RBG, suffix=suffix)
                    save_path = os.path.join(config["MODEL_DIR"], file_name)

                save_meta_data(params=parameters, model_config=model_config, save_path=save_path)

        except Exception as e:
            logger.exception(f"Saving meta data failed: {str(e)}")

    if save_plot:
        try:
            logger.info("Saving plot of validation predictions")
            if isinstance(model, KModels):
                val_indices = model.get_validation_indices()
                val_labels = [data.get("train_labels")[index] for index in val_indices]
                val_predictions = get_k_predictions(model.models, val_indices, data.get("train_images"),
                                                    data.get("train_labels"))
                suffix = "k_fold_pr_curve"
                num_curves = parameters.k
            else:
                val_predictions = model.predict(data.get("val_images"))
                val_labels = data.get("val_labels")
                suffix = "pr_curve"
                num_curves = 1

            filename = format_filename(DataType.PLOT, DataSource.RBG, suffix=suffix)
            save_path = os.path.join(config["PLOT_DIR"], filename)
            plot_precision_recall(num_curves=num_curves, predicted_labels=val_predictions, true_labels=val_labels,
                                  save_path=save_path, show=False)
        except Exception:
            logger.exception("Failed to get validation predictions or save plot")

    if is_cloud_env():
        try:
            logger.info("Uploading temporary log file to storage")
            upload_blob()

            logger.info("Uploading temporary metadata files to storage")
            for i, filepath in enumerate(config["TEMP_META_DATA_FILES"]):
                suffix = f"fold_{models_list[i].model_id}"
                save_filename = format_filename(DataType.METADATA, DataSource.RBG, suffix=suffix)
                destination = os.path.join(config["METADATA_DIR"], save_filename)
                upload_blob(config["BUCKET"], filepath, destination)

        except Exception as e:
            logger.exception(f"Failed to upload temp file or metadata to storage {str(e)}")

    return


if __name__ == '__main__':
    main()
