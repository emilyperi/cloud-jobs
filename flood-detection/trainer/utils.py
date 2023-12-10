from __future__ import annotations

import io
import json
import math
import os.path
from contextlib import redirect_stdout
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from trainer.dtypes import Score, DataType, DataSource, Parameters, ModelConfig, CustomJSONEncoder, FILE_EXT_MAP
from trainer.exceptions import UtilsIOException, UtilsValueException



if TYPE_CHECKING:
    from trainer.model import Model


def get_function_stdout(func):
    stream = io.StringIO()
    with redirect_stdout(stream):
        func()
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def load_json_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except IOError:
        raise UtilsIOException(f"Cannot load config from file {file_path}")
    except json.JSONDecodeError as e:
        raise UtilsValueException(f"Cannot deserialize config file {str(e)}")


def sigmoid(z):
    if np.any(z) > 50 or np.any(z) < -50:
        print(f'z caused overflow {z}')
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def split_indices(indices, labels, test_size=0.2, random_state=23):
    # return order is train_indices, test_indices
    return train_test_split(indices,
                            test_size=test_size,
                            stratify=labels,
                            random_state=random_state)


def plot_changes_in_accuracy(training_accuracy: List[float],
                             validation_accuracy: List[float],
                             parameter_values: np.ndarray,
                             title: str,
                             x_axis_label: str,
                             save_path: str):
    # Ensure that the lengths of provided lists are the same
    if not (len(training_accuracy) == len(validation_accuracy) == len(parameter_values)):
        raise UtilsValueException("The provided lists must have the same length.")

    # Plotting the training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(parameter_values, training_accuracy, label='Training Accuracy', marker='o', linestyle='-')
    plt.plot(parameter_values, validation_accuracy, label='Validation Accuracy', marker='o', linestyle='--')

    # Setting title and labels
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    # Display the plot
    plt.show()


def k_score_summary(scores: List[Score], metric_name: str):
    train_metrics = [s.train_metric for s in scores]
    val_metrics = [s.val_metric for s in scores]
    train_loss = [s.train_loss for s in scores]
    val_loss = [s.val_loss for s in scores]
    summary = (f"\n=================== Metric Summary ===================\n"
               f"Average {metric_name}: Training: {np.mean(train_metrics):.2f}, Validation: {np.mean(val_metrics):.2f}\n"
               f"Standard Deviation of {metric_name}: Training {np.std(train_metrics):.2f}, Validation {np.std(val_metrics):.2f}\n"
               f"Minimum {metric_name}: Training {np.min(train_metrics):.2f}, Validation: {np.min(val_metrics):.2f}\n"
               f"Maximum {metric_name}: Training: {np.max(train_metrics):.2f}, Validation: {np.max(val_metrics):.2f}\n"
               f"=================== Loss Summary ===================\n"
               f"Average Loss: Training: {np.mean(train_loss):.2f}, Validation: {np.mean(val_loss):.2f}\n"
               f"Standard Deviation of Loss: Training {np.std(train_loss):.2f}, Validation {np.std(val_loss):.2f}\n"
               f"Minimum Loss: Training {np.min(train_loss):.2f}, Validation: {np.min(val_loss):.2f}\n"
               f"Maximum Loss: Training: {np.max(train_loss):.2f}, Validation: {np.max(val_loss):.2f}")

    return summary


def plot_precision_recall(num_curves, predicted_labels, true_labels, from_logits=True, save_path=None, show=True):
    plt.figure(figsize=(10, 7) if num_curves > 1 else (8, 6))
    plt.title('Precision-Recall Curves for Multiple Folds' if num_curves > 1 else 'Precision-Recall Curve')

    if num_curves == 1:
        predicted_labels = [predicted_labels]
        true_labels = [true_labels]

    for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
        if from_logits:
            pred = sigmoid(pred)
        precision, recall, _ = precision_recall_curve(true, pred)
        label = f'Fold {i + 1}' if num_curves > 1 else None
        plt.plot(recall, precision, marker='.', label=label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if num_curves > 1:
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()


def format_filename(data_type: DataType, data_source: DataSource, suffix: str = None, ext: bool = True):
    filename = f'{data_source.value}_{data_type.value}'

    if suffix:
        filename = f'{filename}_{suffix}'

    file_ext = FILE_EXT_MAP.get(data_type)
    if ext and file_ext:
        return f'{filename}.{file_ext.value}'
    return filename


def save_models(models: List[Model], data_source: DataSource, base_path: str, ext=False):

    for i, model in enumerate(models):
        suffix = f"fold_{model.model_id}"
        filename = format_filename(DataType.MODEL, data_source, suffix=suffix, ext=ext)
        save_path = os.path.join(base_path, filename)
        try:
            model.save(save_path)
        except IOError:
            raise UtilsIOException(f"Error saving model to save_path {save_path}")


def save_meta_data(params: Parameters, model_config: ModelConfig, save_path: str):
    params_dict = params.serialize()
    mconfig_dict = model_config.serialize()
    meta_data = dict(parameters=params_dict, model_config=mconfig_dict)
    try:
        with open(save_path, 'w') as file:
            json.dump(meta_data, file, indent=4, cls=CustomJSONEncoder)
    except IOError:
        raise UtilsIOException(f"Cannot save data to file {save_path}")
    except TypeError as e:
        raise UtilsValueException(f"Cannot serialize meta data contents {str(e)}")


def display_images(num_images, rgb_images, captions=None, save_path=None, show=True):
    # Calculate the number of rows needed
    rows = math.ceil(num_images / 4)

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(rows, 4, figsize=(15, 5 * rows))

    # Handle the case where there's only one row, which means axs is a 1D array
    if rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(num_images):
        row = i // 4
        col = i % 4
        img = rgb_images[i]
        if captions:
            axs[row, col].set_title(captions[i])
        axs[row, col].imshow(img)
        axs[row, col].axis('off')  # To hide axes

    # Turn off any remaining empty subplots
    for i in range(num_images, rows * 4):
        row = i // 4
        col = i % 4
        axs[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def get_mislabeled_indices(true_labels, predicted_labels):
    return [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]

