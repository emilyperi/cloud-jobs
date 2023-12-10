from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Tuple
import json
import numpy as np
import tensorflow as tf
import math


def set_lr_scheduler(optimizer, batch_size, train_size, epochs):
    warmup_epochs = math.ceil(0.05 * epochs)
    decay_steps = (epochs - warmup_epochs) * (train_size // batch_size)
    warmup_steps = warmup_epochs * (train_size // batch_size)
    lr_config = optimizer.get("config", {}).get("learning_rate")
    if lr_config and isinstance(lr_config, dict):
        lr_config["config"]["warmup_steps"] = tf.constant(warmup_steps, dtype=tf.int32)
        lr_config["config"]["decay_steps"] = tf.constant(decay_steps, dtype=tf.int32)
        lr_config["config"]["initial_learning_rate"] = tf.constant(lr_config["config"]["initial_learning_rate"],
                                                                   dtype=tf.float32)

    return optimizer


@dataclass
class Parameters:
    optimizer: Union[str, tf.keras.optimizers.Optimizer]
    loss: Union[str, tf.keras.losses.Loss]
    metrics: List[Union[str, tf.keras.metrics.Metric]]
    train_size: int = 2368
    batch_size: int = 32
    epochs: int = 10
    k: int = 10

    def serialize(self):
        param_dict = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'k': self.k,
            "train_size": self.train_size
        }

        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer) or isinstance(self.optimizer, tf.keras.optimizers.legacy.Optimizer):
            param_dict['optimizer'] = tf.keras.optimizers.serialize(self.optimizer)
        else:
            param_dict['optimizer'] = self.optimizer

        if isinstance(self.loss, tf.keras.losses.Loss):
            param_dict['loss'] = tf.keras.losses.serialize(self.loss)
        else:
            param_dict["loss"] = self.loss

        metric_serialized = []
        for metric in self.metrics:
            if isinstance(metric, tf.keras.metrics.Metric):
                metric_serialized.append(tf.keras.metrics.serialize(metric))
            else:
                metric_serialized.append(metric)
        param_dict['metrics'] = metric_serialized
        return param_dict

    def metric_key(self):
        primary_metric = self.metrics[0]
        if isinstance(primary_metric, str):
            return primary_metric
        else:
            return primary_metric.get_config()['name']

    @classmethod
    def deserialize(cls, config: dict):
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 10)
        k = config.get('k', 10)
        loss = config.get('loss')
        metrics = config.get('metrics')
        optimizer = config.get('optimizer')
        train_size = config.get("train_size")
        if not optimizer:
            raise ValueError('optimizer parameter required')
        if not metrics:
            raise ValueError('metric parameter required')
        if not loss:
            raise ValueError('loss parameter required')

        if isinstance(loss, dict):
            loss = tf.keras.losses.deserialize(loss)

        if isinstance(optimizer, dict):
            optimizer = set_lr_scheduler(optimizer, batch_size, train_size, epochs)
            optimizer = tf.keras.optimizers.deserialize(optimizer)

        for i in range(len(metrics)):
            if isinstance(metrics[i], dict):
                metrics[i] = tf.keras.metrics.deserialize(metrics[i])

        return cls(optimizer=optimizer, loss=loss, metrics=metrics, k=k, epochs=epochs, batch_size=batch_size, train_size=train_size)


@dataclass
class Score:
    train_loss: float
    train_metric: float
    val_loss: float
    val_metric: float


@dataclass
class ModelConfig:
    bottom_layers: List[tf.keras.layers.Layer]
    top_layers: List[tf.keras.layers.Layer]
    input_shape: Tuple[int, int, int] = (416, 416, 2)
    train_size = 2368
    saved_model: str = None
    data_augmentation: bool = False

    def serialize(self):
        config_dict = {
            "input_shape": self.input_shape,
            "saved_model": self.saved_model,
            "data_augmentation": self.data_augmentation
        }

        bottom_layers_serialized = []
        for layer in self.bottom_layers:
            bottom_layers_serialized.append(tf.keras.layers.serialize(layer))

        config_dict['bottom_layers'] = bottom_layers_serialized

        top_layers_serialized = []
        for layer in self.top_layers:
            top_layers_serialized.append(tf.keras.layers.serialize(layer))

        config_dict['top_layer'] = top_layers_serialized

        return config_dict

    @classmethod
    def deserialize(cls, config):
        input_shape = tuple(config.get('input_shape'))
        bottom_layers = config.get('bottom_layers')
        top_layers = config.get('top_layers')
        saved_model = config.get("saved_model")
        data_augmentation = config.get("data_augmentation", False)
        if not bottom_layers:
            raise ValueError('bottom layers parameter required')

        if not top_layers:
            raise ValueError('top layers parameter required')

        bottom_layers = [tf.keras.layers.deserialize(layer) for layer in bottom_layers]
        top_layers = [tf.keras.layers.deserialize(layer) for layer in top_layers]

        return cls(input_shape=input_shape, bottom_layers=bottom_layers, top_layers=top_layers, saved_model=saved_model,
                   data_augmentation=data_augmentation)


class ModelType(Enum):
    K_FOLD = "k_fold"
    SINGLE = "single"


class PlotType(Enum):
    PR_CURVE = "pr_curve"


class ModelTask(Enum):
    PLOT = "plot"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    TRAIN = "train"
    TUNE = "tune"
    PROCESS_IMAGES = "process_images"
    SPLIT_DATA = "split_data"
    LOAD_MODEL = "load_model"
    SAVE_MODEL = "save_model"
    SAVE_METADATA = "save_metadata"


class DataType(Enum):
    MODEL = "model"
    WEIGHTS = "weights"
    NDARRAY = "np"
    DATAFRAME = "df"
    PLOT = "plot"
    METADATA = "meta_data"
    IMAGE = "image"


class DataSource(Enum):
    RBG = "rgb"
    MULTI = "multi"
    SAR = "sar"


class FileExt(Enum):
    H5 = "h5"
    NPY = "npy"
    JSON = "json"
    PNG = "png"
    KERAS = "keras"


FILE_EXT_MAP = {
    DataType.MODEL: FileExt.KERAS,
    DataType.WEIGHTS: FileExt.H5,
    DataType.METADATA: FileExt.JSON,
    DataType.NDARRAY: FileExt.NPY,
    DataType.PLOT: FileExt.PNG,
    DataType.IMAGE: FileExt.PNG
}


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        # Add more custom handlers here if needed
        return json.JSONEncoder.default(self, obj)
