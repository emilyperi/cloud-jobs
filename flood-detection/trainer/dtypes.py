from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Tuple
import json
import numpy as np
import tensorflow as tf


@dataclass
class Parameters:
    optimizer: Union[str, tf.keras.optimizers.Optimizer]
    loss: Union[str, tf.keras.losses.Loss]
    metrics: List[Union[str, tf.keras.metrics.Metric]]
    batch_size: int = 32
    epochs: int = 10
    k: int = 10

    def serialize(self):
        param_dict = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'k': self.k
        }

        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
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
        if not optimizer:
            raise ValueError('optimizer parameter required')
        if not metrics:
            raise ValueError('metric parameter required')
        if not loss:
            raise ValueError('loss parameter required')

        if isinstance(loss, dict):
            loss = tf.keras.losses.deserialize(loss)

        if isinstance(optimizer, dict):
            optimizer = tf.keras.optimizers.deserialize(optimizer)

        for i in range(len(metrics)):
            if isinstance(metrics[i], dict):
                metrics[i] = tf.keras.metrics.deserialize(metrics[i])

        return cls(optimizer=optimizer, loss=loss, metrics=metrics, k=k, epochs=epochs, batch_size=batch_size)


@dataclass
class Score:
    train_loss: float
    train_metric: float
    val_loss: float
    val_metric: float


@dataclass
class ModelConfig:
    layers: List[tf.keras.layers.Layer]
    input_shape: Tuple[int, int, int] = (512, 512, 3)

    def serialize(self):
        config_dict = {
            "input_shape": self.input_shape
        }

        layers_serialized = []
        for layer in self.layers:
            layers_serialized.append(tf.keras.layers.serialize(layer))

        config_dict['layers'] = layers_serialized
        return config_dict

    @classmethod
    def deserialize(cls, config):
        input_shape = tuple(config.get('input_shape', (512, 512, 3)))
        layers = config.get('layers')
        if not layers:
            raise ValueError('layers parameter required')

        layers = [tf.keras.layers.deserialize(layer) for layer in layers]

        return cls(input_shape=input_shape, layers=layers)


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