import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from trainer import config
from trainer.dtypes import Parameters, ModelConfig, DataSource, DataType
from trainer.model import KModels
from trainer.logging import logger, upload_blob
from trainer.utils import split_indices, save_models, create_save_path, save_meta_data, get_k_predictions, \
    display_images, \
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
        '--make-predictions',
        action='store_true',
        help='Make model predictions'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Train model'
    )
    parser.add_argument(
        '--skip-save',
        action='store_true',
        help='Make model predictions'
    )

    return parser.parse_args()


def get_config_file_path(config_name):
    if config_name:
        package_dir = Path(__file__).parent  # Path to the directory containing this file
        config_path = package_dir / 'configs' / config_name
        return config_path


def main():
    args = get_args()
    k = args.k
    batch_size = args.batch_size
    epochs = args.epochs
    model_config_file = get_config_file_path(args.model_config)
    parameter_config_file = get_config_file_path(args.parameter_config)
    make_predictions = args.make_predictions
    skip_train = args.skip_train
    skip_save = args.skip_save

    try:
        logger.info("Loading data")
        labels = np.load(config["LABELS_FILE_PATH"])
        images = np.load(config["IMG_FILE_PATH"])
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
    except Exception:
        logger.exception("Failed to preprocess or split data into train and test sets")
        return

    parameters = None
    model_config = None
    k_fold_model = None
    try:
        logger.info("Initializing model")
        if model_config_file:
            model_config = ModelConfig.deserialize(config=load_json_config(model_config_file))
        else:
            model_config = ModelConfig(layers=[Flatten(),
                                               Dense(256, activation='relu'),
                                               Dense(1)],
                                       input_shape=(512, 512, 3))
        if parameter_config_file:
            parameters = Parameters.deserialize(config=load_json_config(parameter_config_file))
        else:
            parameters = Parameters(batch_size=batch_size,
                                    epochs=epochs,
                                    optimizer=Adam(),
                                    loss=BinaryCrossentropy(from_logits=True),
                                    metrics=[BinaryAccuracy()],
                                    k=k)

        k_fold_model = KModels(k=parameters.k)
        k_fold_model.init_models(model_config)

    except Exception:
        logger.exception(f"Failed to initialize k model with params {parameters}, model configuration "
                         f"{model_config}: model: {k_fold_model} ")
        return

    if not skip_train:
        try:
            logger.info("Beginning Training")
            k_fold_model.kfold_train(train_data=(train_images, train_labels), params=parameters)
            logger.info(k_score_summary(scores=k_fold_model.scores, metric_name=parameters.metric_key()))
        except Exception:
            logger.exception("Training failed")
            return

    if not skip_save:
        try:
            logger.info("Saving Models")
            save_models(models=k_fold_model.models, data_source=DataSource.RBG, base_path=config["MODEL_DIR"])
        except Exception as e:
            logger.exception(f"Saving model failed: {str(e)}")

        try:
            logger.info("Saving Model Meta Data")
            for i in range(len(k_fold_model.models)):
                save_meta_data(params=parameters, model_config=model_config, data_source=DataSource.RBG,
                               model_id=str(i),
                               base_path=config["MODEL_DIR"])
        except Exception as e:
            logger.exception(f"Saving meta data failed: {str(e)}")

    if make_predictions:
        try:
            logger.info("Saving plot of validation predictions")
            val_indices = k_fold_model.get_validation_indices()
            val_predictions, val_labels = get_k_predictions(k_fold_model.models, val_indices, train_images,
                                                            train_labels)

            save_path = create_save_path(DataType.PLOT, DataSource.RBG, base_path=config["PLOT_DIR"],
                                         suffix="k_fold_pr_curve", )
            plot_precision_recall(num_curves=parameters.k, predicted_labels=val_predictions, true_labels=val_labels,
                                  save_path=save_path, show=False)
        except Exception:
            logger.exception("Failed to get validation predictions or save plot")

    if config["TRAIN_ENV"].lower() == 'cloud':
        try:
            logger.info("Saving temporary log file to storage")
            upload_blob()
        except Exception as e:
            logger.exception(f"Failed to upload temp file to storage {str(e)}")

    return


if __name__ == '__main__':
    main()
