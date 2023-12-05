import os.path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential

from trainer.env import TENSORBOARD_DIR, CHECKPOINT_DIR, CHECKPOINT_TEMPLATE, GPU_ENABLED
from trainer.dtypes import Parameters, Score, ModelConfig
from trainer.exceptions import UninitializedModelError, UntrainedModelError, KModelValueError
from trainer.utils import get_function_stdout
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
                          ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch')]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, config: ModelConfig):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=config.input_shape)

        # Make base layers non-trainable
        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential([base_model, *config.layers])
        logger.debug(f"Model Summary {get_function_stdout(model.summary)}")
        self._model = model
        self._base_model = base_model

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
            strategy = tensorflow.distribute.MirroredStrategy()
            with strategy.scope():
                self._model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)
        else:
            self._model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)

        logger.debug(f"checkpoint dir is {self.checkpoint_dir}")
        if tensorflow.io.gfile.exists(self.checkpoint_dir):
            latest_checkpoint = tensorflow.train.latest_checkpoint(self.checkpoint_dir)
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

    def predict(self, test_images):
        if not self._model:
            raise UninitializedModelError("Cannot make predictions with un-initialized model")

        return self._model.predict(test_images)

    def binary_predict(self, test_images, threshold=0.5):
        if not self._model:
            raise UninitializedModelError("Cannot make predictions with un-initialized model")

        predictions = self.predict(test_images)
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
            raise KModelValueError(f"Number of weights {weights_paths} does not match number of models {len(self.models)}")

        for i, model in enumerate(self.models):
            model.load_weights(weights_paths[i])

    def init_split(self, k, train_images, train_labels, random_state, save=True):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        split_indices = list(skf.split(train_images, train_labels))
        if save:
            self.split = split_indices

        return split_indices

    def kfold_train(self, train_data: Tuple, params: Parameters, random_state=13):
        train_images, train_labels = train_data
        splits = self.init_split(params.k, train_images, train_labels, random_state)

        for i, (train_index, val_index) in enumerate(splits):
            logger.debug(f'Beginning Fold {i}')
            model = self.models[i]
            model.train(train_data=(train_images[train_index], train_labels[train_index]),
                        val_data=(train_images[val_index], train_labels[val_index]),
                        params=params)

            self.scores.append(model.get_score(metric_key=params.metric_key()))

    def ensemble_predict(self, test_images, threshold=0.5):
        sum_k_pred = np.zeros((test_images.shape[0],))
        for model in self.models:
            pred = model.predict(test_images).squeeze()
            sum_k_pred += pred

        averaged_pred = sum_k_pred / self.k
        binary_labels = np.where(averaged_pred > threshold, 1, 0)

        return averaged_pred, binary_labels

    def get_validation_indices(self):
        if not self.split:
            raise KModelValueError("Uninitialized split for training and validation indices")
        return [indices[1] for indices in self.split]

    def get_training_indices(self):
        if not self.split:
            raise KModelValueError("Uninitialized split for training and validation indices")
        return [indices[0] for indices in self.split]
