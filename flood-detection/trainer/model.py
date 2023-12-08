import os.path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Input

from trainer.env import TENSORBOARD_DIR, CHECKPOINT_DIR, CHECKPOINT_TEMPLATE, GPU_ENABLED
from trainer.dtypes import Parameters, Score, ModelConfig
from trainer.exceptions import UninitializedModelError, UntrainedModelError, KModelValueError
from trainer.utils import get_function_stdout, sigmoid
from trainer.logging import logger


class Model:
    log_dir = TENSORBOARD_DIR
    base_check_dir = CHECKPOINT_DIR
    check_template = CHECKPOINT_TEMPLATE
    gpu_enabled = GPU_ENABLED

    def __init__(self, model_id):
        self._model = None
        self._base_model = None
        self.history = None
        self.model_id = model_id

        self.tensorboard_dir = os.path.join(Model.log_dir, f"fold_{model_id}")
        self.checkpoint_dir = os.path.join(Model.base_check_dir, f"fold_{model_id}")
        checkpoint_path = os.path.join(self.checkpoint_dir, Model.check_template)

        self.callbacks = [TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1),
                          ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch'),
                          EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, config: ModelConfig):
        if config.saved_model:
            model, base_model = self.load_model(config.saved_model)
        else:
            model, base_model = self.create_model(config)
        logger.debug(f"Model {self.model_id} Summary {get_function_stdout(model.summary)}")
        self._model = model
        self._base_model = base_model

    def create_model(self, config: ModelConfig):
        image_input = Input(shape=config.input_shape, name='image_input')

        # Data augmentation layers (if required)
        if config.data_augmentation:
            logger.debug("Adding data augmentation layers")
            x = tf.keras.layers.RandomFlip("horizontal")(image_input)
            x = tf.keras.layers.RandomRotation(0.1)(x)
            x = tf.keras.layers.RandomZoom(0.1)(x)
        else:
            x = image_input

        # ResNet50 as the base model
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
        base_model.trainable = False

        # Adding the base model
        x = base_model(x)

        for layer in config.layers:
            x = layer(x)

        # Create the model
        model = tf.keras.models.Model(inputs=image_input, outputs=x)

        return model, base_model

    def load_model(self, load_path):
        model = tf.keras.models.load_model(load_path.format(self.model_id))
        base_model = model.get_layer(index=0) if hasattr(model, 'get_layer') else None
        return model, base_model

    def unfreeze_layers(self, n_layers):
        if not self._base_model:
            raise UninitializedModelError("Cannot unfreeze layers on un-initialized model")
        trainable_index = len(self._base_model.layers) - n_layers
        for layer in self._base_model.layers[trainable_index:]:  # Unfreeze the last n layers
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    def train(self, train_data: Tuple, val_data: Tuple, params: Parameters):
        if not self._model:
            raise UninitializedModelError("Cannot train an un-initialized model")

        if self.gpu_enabled:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self._model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)
        else:
            self._model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)

        if tf.io.gfile.exists(self.checkpoint_dir):
            logger.debug(f"Loading from checkpoint {self.checkpoint_dir}")
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.load_weights(latest_checkpoint)

        train_images, train_labels = train_data
        val_images, val_labels = val_data
        history = self._model.fit(train_images,
                                  train_labels,
                                  batch_size=params.batch_size,
                                  epochs=params.epochs,
                                  validation_data=(val_images, val_labels),
                                  callbacks=self.callbacks)

        self.history = history.history

    def predict(self, test_images, apply_sigmoid=True):
        if not self._model:
            raise UninitializedModelError("Cannot make predictions with un-initialized model")

        if apply_sigmoid:
            return sigmoid(self._model.predict(test_images))

        return self._model.predict(test_images)

    def binary_labels(self, predictions, threshold=0.5):
        if not self._model:
            raise UninitializedModelError("Cannot make predictions with un-initialized model")

        return np.where(predictions > threshold, 1, 0)

    def save(self, save_path):
        if not self._model:
            raise UninitializedModelError("Cannot save un-initialized model")

        logger.debug(f"Model save_path {save_path}")
        self._model.save(save_path)

    def load_weights(self, weights_path):
        if not self._model:
            raise UninitializedModelError("Cannot load weights for un-initialized model")

        self._model.load_weights(weights_path)

    def evaluate(self, test_images, test_labels):
        if not self._model:
            raise UninitializedModelError("Cannot load model for un-initialized model")

        return self._model.v(test_images, test_labels)

    def get_score(self, metric_key):
        if not self.history:
            raise UntrainedModelError("Cannot lookup training history on untrained model")

        return Score(train_loss=self.history['loss'][-1],
                     train_metric=self.history[metric_key][-1],
                     val_loss=self.history[f'val_loss'][-1],
                     val_metric=self.history[f'val_{metric_key}'][-1])


class KModels:
    def __init__(self, k: int):
        if k < 2:
            raise KModelValueError("Number of folds k must be greater than 1")
        self.k = k
        self.models = [Model(model_id=i) for i in range(k)]
        self.scores: List[Score] = []
        self.split = None

    def init_models(self, config: ModelConfig):
        for model in self.models:
            model.model = config

    def load_weights(self, weights_paths: List[str]):
        if len(weights_paths) != len(self.models):
            raise KModelValueError(
                f"Number of weights {weights_paths} does not match number of models {len(self.models)}")

        for i, model in enumerate(self.models):
            model.load_weights(weights_paths[i])

    def evaluate_models(self, test_images, test_labels):
        losses = []
        metrics = []

        for model in self.models:
            loss, metric = model.evaluate(test_images, test_labels)
            losses.append(loss)
            metrics.append(metric)

        return losses, metrics

    def init_split(self, train_images, train_labels, random_state=13):
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=random_state)
        split_indices = list(skf.split(train_images, train_labels))
        self.split = split_indices

        return split_indices

    def kfold_train(self, train_data: Tuple, params: Parameters):
        train_images, train_labels = train_data
        splits = self.split

        for i, (train_index, val_index) in enumerate(splits):
            logger.debug(f'Beginning Fold {i}')
            model = self.models[i]
            model.train(train_data=(train_images[train_index], train_labels[train_index]),
                        val_data=(train_images[val_index], train_labels[val_index]),
                        params=params)

            self.scores.append(model.get_score(metric_key=params.metric_key()))

    def ensemble_predict(self, test_images, apply_sigmoid=True):
        sum_k_pred = np.zeros((test_images.shape[0],))
        for model in self.models:
            pred = model.predict(test_images, apply_sigmoid=apply_sigmoid).squeeze()
            sum_k_pred += pred

        averaged_pred = sum_k_pred / self.k

        return averaged_pred

    def binary_labels(self, predictions, threshold):
        binary_labels = []
        for i, model in enumerate(self.models):
            binary_labels.append(model.binary_labels(predictions[i], threshold=threshold))

        return binary_labels

    def val_predictions(self, images, labels, apply_sigmoid=True):
        predictions = []
        true_labels = []
        for i, (train_index, val_index) in enumerate(self.split):
            logger.debug(f'Beginning predictions for model {i}')
            model = self.models[i]
            predictions.append(model.predict(images[val_index], apply_sigmoid=apply_sigmoid))
            true_labels.append(labels[val_index])
        return predictions, true_labels

    def train_predictions(self, images, labels, apply_sigmoid=True):
        predictions = []
        true_labels = []
        for i, (train_index, val_index) in enumerate(self.split):
            logger.debug(f'Beginning predictions for model {i}')
            model = self.models[i]
            predictions.append(model.predict(images[train_index], apply_sigmoid=apply_sigmoid))
            true_labels.append(labels[train_index])
        return predictions, true_labels

    def test_predictions(self, test_images, apply_sigmoid=True):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(test_images, apply_sigmoid=apply_sigmoid))
        return predictions

    def get_validation_indices(self):
        if not self.split:
            raise KModelValueError("Uninitialized split for training and validation indices")
        return [indices[1] for indices in self.split]

    def get_training_indices(self):
        if not self.split:
            raise KModelValueError("Uninitialized split for training and validation indices")
        return [indices[0] for indices in self.split]
