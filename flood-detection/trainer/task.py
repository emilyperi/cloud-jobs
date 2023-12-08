import argparse
import os.path
from pathlib import Path
import tempfile
from typing import List, Union, Dict

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from trainer.env import config
from trainer.dtypes import Parameters, ModelConfig, DataSource, DataType, ModelType, ModelTask, PlotType
from trainer.model import KModels, Model
from trainer.logging import logger, upload_blob
from trainer.utils import split_indices, save_models, format_filename, save_meta_data, \
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
        '--threshold',
        type=float,
        default=0.5,
        help='Number epochs for training models'
    )
    parser.add_argument(
        '--num-unfreeze-layers',
        type=int,
        default=10,
        help='Number layers to unfreeze'
    )
    parser.add_argument(
        '--prediction-type',
        type=str,
        default='validation',
        help='Number layers to unfreeze'
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
        '--plot',
        action='store_true',
        help='Save a PR Curve plot of the validation predictions'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Skips training the model'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the model'
    )
    parser.add_argument(
        '--save-metadata',
        action='store_true',
        help='Save the model meta data'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Makes predictions on the loaded model(s)'
    )
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='Skips saving the model'
    )
    parser.add_argument(
        '--split-train-test',
        action='store_true',
        help='Skips saving the model'
    )
    parser.add_argument(
        '--process-images',
        action='store_true',
        help='Skips saving the model'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Makes predictions on the loaded model(s)'
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Makes predictions on the loaded model(s)'
    )
    parser.add_argument(
        '--data-augmentation',
        action='store_true',
        help='Makes predictions on the loaded model(s)'
    )

    return parser.parse_args()


def get_config_file_path(config_name):
    if config_name:
        package_dir = Path(__file__).parent  # Path to the directory containing this file
        config_path = package_dir / 'configs' / config_name
        return config_path


def log_summary(model: Union[Model, KModels], metric_key):
    if isinstance(model, KModels):
        logger.info(k_score_summary(scores=model.scores, metric_name=metric_key))
    else:
        score = model.get_score(metric_key)
        logger.info(f'Model Summary:\n'
                    f'Train Loss: {score.train_loss}\n'
                    f'Validation Loss: {score.val_loss}\n'
                    f'Train Metric ({metric_key}): {score.train_metric}\n'
                    f'Validation Metric ({metric_key}): {score.val_metric}')


def get_models_list(model):
    if isinstance(model, KModels):
        return model.models
    else:
        return [model]


def initialize_parameters(k, batch_size, epochs, config_file):
    logger.info("Initializing model parameters")
    if config_file:
        parameters = Parameters.deserialize(config=load_json_config(config_file))
    else:
        parameters = Parameters(batch_size=batch_size,
                                epochs=epochs,
                                optimizer=Adam(),
                                loss=BinaryCrossentropy(from_logits=True),
                                metrics=[BinaryAccuracy()],
                                k=k)
    return parameters


def initialize_model_config(input_shape, config_file, saved_model=None, data_augmentation=False):
    logger.info("Initializing model configuration")

    if config_file:
        model_config = ModelConfig.deserialize(config=load_json_config(config_file))
    else:
        model_config = ModelConfig(layers=[Flatten(), Dense(256, activation='relu'), Dense(1)], input_shape=input_shape)

    if saved_model:
        model_config.saved_model = saved_model

    if data_augmentation:
        model_config.data_augmentation = True

    return model_config


def load_data(model_tasks: List[ModelTask]):
    val_labels, val_images = None, None
    test_labels, test_images = None, None

    logger.info("Loading training data")
    train_labels = np.load(config["TRAIN_LABELS_FILE_PATH"])
    train_images = np.load(config["TRAIN_IMG_FILE_PATH"])
    if config.get("VAL_LABELS_FILE_PATH"):
        val_labels = np.load(config["VAL_LABELS_FILE_PATH"])
    if config.get("VAL_IMG_FILE_PATH"):
        val_images = np.load(config["VAL_IMG_FILE_PATH"])

    if ModelTask.EVALUATE in model_tasks:
        logger.info("Loading test data")
        if config.get("TEST_LABELS_FILE_PATH"):
            test_labels = np.load(config["TEST_LABELS_FILE_PATH"])
        if config.get("TEST_IMG_FILE_PATH"):
            test_images = np.load(config["TRAIN_IMG_FILE_PATH"])

    if ModelTask.SPLIT_TRAIN_TEST in model_tasks:
        logger.info("Splitting data into train and test sets")
        indices = np.arange(len(train_images))
        train_indices, test_indices = split_indices(indices, train_labels)
        train_images, test_images = train_images[train_indices], train_images[test_indices]
        train_labels, test_labels = train_labels[train_indices], train_labels[test_indices]

    if ModelTask.PROCESS_IMAGES in model_tasks:
        train_images = preprocess_input(train_images)
        if test_images:
            test_images = preprocess_input(test_images)

    data = dict(train_images=train_images, val_images=val_images, test_images=test_images,
                train_labels=train_labels, val_labels=val_labels, test_labels=test_labels)

    return data


def get_predictions(data, model, prediction_type, threshold):
    if prediction_type == 'train':
        if isinstance(model, KModels):
            predictions, true_labels = model.train_predictions(data["train_images"], data["train_labels"])
        else:
            predictions = model.predict(data["train_images"])
            true_labels = data.get("train_labels")
    elif prediction_type == 'validation':
        if isinstance(model, KModels):
            predictions, true_labels = model.val_predictions(data["train_images"], data["train_labels"])
        else:
            predictions = model.predict(data["val_images"])
            true_labels = data.get("val_labels")
    elif prediction_type == 'test':
        if isinstance(model, KModels):
            predictions = model.test_predictions(data["test_images"])
        else:
            predictions = model.predict(data.get("test_images"))
        true_labels = data.get("test_labels")
    else:
        assert isinstance(model, KModels)
        true_labels = data.get("test_labels")
        predictions = model.ensemble_predict(data.get("test_images"))

    predicted_labels = model.binary_labels(predictions, threshold=threshold)

    return predictions, predicted_labels, true_labels


def initialize_model(k: int, model_id: int, model_config: ModelConfig, model_type: ModelType):
    if model_type == ModelType.K_FOLD:
        model = KModels(k=k)
        model.init_models(model_config)
    else:
        model = Model(model_id=model_id)
        model.model = model_config
    return model


def create_plot(model: Union[Model, KModels], data: Dict, plot_type: PlotType, prediction_type: str, threshold: float):
    logger.info("Saving plot of validation predictions")
    predictions, _, true_labels = get_predictions(data=data, model=model, prediction_type=prediction_type,
                                                  threshold=threshold)

    if isinstance(model, KModels):
        suffix = f"k_fold_{plot_type.value}"
        num_curves = model.k
    else:
        suffix = f"fold_{model.model_id}_{plot_type.value}"
        num_curves = 1

    filename = format_filename(DataType.PLOT, DataSource.RBG, suffix=suffix)
    save_path = os.path.join(config["PLOT_DIR"], filename)
    plot_precision_recall(num_curves=num_curves, predicted_labels=predictions, true_labels=true_labels,
                          save_path=save_path, show=False)


def get_load_dir(model_list: List[Model], model_dir: str, data_source: DataSource, is_cloud_env: bool):
    load_paths = []
    for model in model_list:
        ext = False if is_cloud_env else True
        suffix = f"fold_{model.model_id}"
        path = format_filename(data_type=DataType.MODEL, data_source=data_source, suffix=suffix, ext=ext)
        full_path = os.path.join(model_dir, path)
        load_paths.append(full_path)
    return load_paths


def main():
    args = get_args()
    k = args.k
    batch_size = args.batch_size
    epochs = args.epochs
    threshold = args.threshold
    num_unfreeze_layers = args.num_unfreeze_layers
    prediction_type = args.prediction_type
    model_config_file = get_config_file_path(args.model_config)
    parameter_config_file = get_config_file_path(args.parameter_config)
    data_augmentation = args.data_augmentation

    model_tasks = []
    if args.plot:
        model_tasks.append(ModelTask.PLOT)
    if args.train:
        model_tasks.append(ModelTask.TRAIN)
    if args.evaluate:
        model_tasks.append(ModelTask.EVALUATE)
    if args.predict:
        model_tasks.append(ModelTask.PREDICT)
    if args.load_model:
        model_tasks.append(ModelTask.LOAD_MODEL)
    if args.split_train_test:
        model_tasks.append(ModelTask.SPLIT_TRAIN_TEST)
    if args.process_images:
        model_tasks.append(ModelTask.PROCESS_IMAGES)
    if args.fine_tune:
        model_tasks.append(ModelTask.TUNE)
    if args.save_model:
        model_tasks.append(ModelTask.SAVE_MODEL)
    if args.save_metadata:
        model_tasks.append(ModelTask.SAVE_METADATA)

    if config["MODEL_TYPE"].lower() == ModelType.K_FOLD.value:
        model_type = ModelType.K_FOLD
    else:
        model_type = ModelType.SINGLE

    plot_type = PlotType.PR_CURVE

    is_cloud_env = config["TRAIN_ENV"].lower() == "cloud"

    parameters = None
    model_config = None

    try:
        saved_model = None
        if ModelTask.LOAD_MODEL in model_tasks:
            saved_model = os.path.join(config["MODEL_LOAD_DIR"], config["SAVED_MODEL_TEMPLATE"])
        model_config = initialize_model_config(input_shape=(512, 512, 3), config_file=model_config_file,
                                               saved_model=saved_model, data_augmentation=data_augmentation)
        parameters = initialize_parameters(k=k, batch_size=batch_size, epochs=epochs, config_file=parameter_config_file)
    except Exception:
        logger.exception(f"Failed to initialize model params: {parameters}, model configuration: {model_config}")
        return

    try:
        data = load_data(model_tasks)
    except IOError:
        logger.exception("Failed to load data")
        return
    except Exception as e:
        logger.exception(f"Failed to load data {str(e)}")
        return

    try:
        model = initialize_model(k=k, model_id=config["MODEL_ID"], model_config=model_config, model_type=model_type)
        if model_type == ModelType.K_FOLD:
            model.init_split(train_images=data["train_images"], train_labels=data["train_labels"])
            data["val_indices"] = model.get_validation_indices()
            data["train_indices"] = model.get_training_indices()
    except Exception as e:
        logger.exception(f"Failed to initialize the model {str(e)}")
        return

    if ModelTask.TRAIN in model_tasks or ModelTask.TUNE in model_tasks:
        try:
            if ModelTask.TUNE:
                models_list = get_models_list(model)
                for model in models_list:
                    model.unfreeze_layers(n_layers=num_unfreeze_layers)

            train_data = (data.get("train_images"), data.get("train_labels"))
            if isinstance(model, KModels):
                model.kfold_train(train_data=train_data, params=parameters)
            else:
                val_data = (data.get("val_images"), data.get("val_labels"))
                model.train(train_data=train_data, val_data=val_data, params=parameters)
        except Exception as e:
            logger.exception(f"Failed to train model {str(e)}")
            return

    if ModelTask.SAVE_MODEL in model_tasks:
        try:
            models_list = get_models_list(model)
            base_path = config["MODEL_SAVE_DIR"]
            save_models(models=models_list, data_source=DataSource.RBG, base_path=base_path, env=config["TRAIN_ENV"])
        except Exception as e:
            logger.exception(f"Saving model failed: {str(e)}")

    if ModelTask.SAVE_METADATA in model_tasks:
        try:
            logger.info("Saving Model Meta Data")
            models_list = get_models_list(model)
            for m in models_list:
                if is_cloud_env:
                    save_path = tempfile.NamedTemporaryFile(mode='w', delete=False).name
                    config["TEMP_META_DATA_FILES"].append(save_path)
                else:
                    suffix = f"fold_{m.model_id}"
                    file_name = format_filename(DataType.METADATA, DataSource.RBG, suffix=suffix)
                    base_path = config["METADATA_DIR"]
                    save_path = os.path.join(base_path, file_name)

                save_meta_data(params=parameters, model_config=model_config, save_path=save_path)
        except Exception as e:
            logger.exception(f"Saving meta data failed: {str(e)}")

    if ModelTask.PREDICT in model_tasks:
        try:
            predictions, binary_labels, true_labels = get_predictions(data=data, prediction_type=prediction_type, threshold=threshold)
            logger.info(f"Predictions: {predictions}, Predicted Labels: {binary_labels}")
        except Exception as e:
            logger.exception(f"Predictions failed {str(e)}")

    if ModelTask.PLOT in model_tasks:
        try:
            create_plot(model=model, data=data, plot_type=plot_type, prediction_type=prediction_type,
                        threshold=threshold)
        except Exception as e:
            logger.exception(f"Failed to get validation predictions or save plot {str(e)}")

    if ModelTask.EVALUATE in model_tasks:
        try:
            test_images = data["test_images"]
            test_labels = data["test_labels"]
            if model_type == ModelType.K_FOLD:
                loss, metric = model.evaluate_models(test_images=test_images, test_labels=test_labels)
            else:
                loss, metric = model.evaluate(test_images=test_images, test_labels=test_labels)
            logger.info(f"Test Loss: {loss}, Test Metric: {metric}")
        except Exception as e:
            logger.exception(f"Evaluate model failed {str(e)}")

    if is_cloud_env:
        try:
            logger.info("Uploading temporary log file to storage")
            upload_blob()

            logger.info("Uploading temporary metadata files to storage")
            models_list = get_models_list(model)
            for i, filepath in enumerate(config["TEMP_META_DATA_FILES"]):
                suffix = f"fold_{models_list[i].model_id}"
                save_filename = format_filename(DataType.METADATA, DataSource.RBG, suffix=suffix)
                base_path = config["METADATA_DIR"]
                destination = os.path.join(base_path, save_filename)
                upload_blob(config["BUCKET"], filepath, destination)

        except Exception as e:
            logger.exception(f"Failed to upload temp file or metadata to storage {str(e)}")

    return


if __name__ == '__main__':
    main()
