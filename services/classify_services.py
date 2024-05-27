import json
import logging
import pathlib
import numpy as np
import tensorflow as tf

from typing import List, Tuple

def load_class_names(labels_file: str) -> List[str]:
    """
    Load class names from a JSON file.

    Args:
    - labels_file: A string representing the path to the JSON file containing class names.

    Returns:
    - A list of strings, where each string is a class name extracted from the file.
    """
    try:
        with open(labels_file, 'r', encoding='utf-8') as file:
            class_names = json.load(file)
        return class_names
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file '{labels_file}' not found.")
    except json.JSONDecodeError:
        raise RuntimeError(f"Error decoding JSON from '{labels_file}'.")
    except Exception as e:
        raise RuntimeError(f"Error loading class names from '{labels_file}': {e}")

def preprocess_image(image_path: str, img_height: int, img_width: int) -> tf.Tensor:
    """
    Preprocess the image to the required size and format.

    Args:
        image_path (str): Path to the image file.
        img_height (int): Height to which the image should be resized.
        img_width (int): Width to which the image should be resized.

    Returns:
        tf.Tensor: Processed image tensor ready for model input.

    Raises:
        FileNotFoundError: If the image file is not found.
        RuntimeError: If an error occurs during image preprocessing.
    """
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        return tf.expand_dims(img_array, 0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image '{image_path}': {e}")
    
def classify_image(model: tf.keras.Model, img_array: np.ndarray, class_names: List[str], top_k: int) -> Tuple[List[str], List[float]]:
    """
    Classify the class of the input image using the trained model.

    Args:
    - model: A TensorFlow model that is already trained.
    - img_array: A preprocessed image array ready for model input.
    - class_names: A list of class names corresponding to the model's output.
    - top_k: An integer representing the number of top classes to return (default is 3).

    Returns:
    - top_classes: A list of class names with the highest confidence scores.
    - top_confidences: A list of confidence percentages corresponding to the top_classes.
    """
    try:
        classifications = model.predict(img_array)
        scores = tf.nn.softmax(classifications[0])
        top_indices = np.argsort(scores)[-top_k:][::-1]

        top_classes = [class_names[idx] for idx in top_indices]
        top_confidences = [100 * scores[idx] for idx in top_indices]

        return top_classes, top_confidences
    except (tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError) as e:
        logging.error(f"Failed to classify image: {e}")
        raise RuntimeError(f"Failed to classify image: {e}")
    
def load_model(model_name: str, path: str) -> tf.keras.Model:
    """
    Load a TensorFlow model from a specified directory and model name.

    Args:
        model_name (str): The name of the model to load.
        path (str): The directory where the model files are stored.

    Returns:
        tf.keras.Model: The loaded TensorFlow model object.

    Raises:
        FileNotFoundError: If the model file is not found.
        RuntimeError: If any other error occurs during the loading process.
    """
    try:
        model = tf.keras.models.load_model(pathlib.Path(path) / f"{model_name}.keras")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{path}/{model_name}.keras' not found.")
    except Exception as e:
        raise RuntimeError(f"Error loading model '{path}/{model_name}.keras': {e}")
    
    return model

def load_labels(model_name: str, path: str) -> pathlib.Path:
    """
    Constructs the path to a JSON file containing labels associated with a model.

    Args:
        model_name (str): The name of the model.
        path (str): The directory path where the model's associated files are stored.

    Returns:
        pathlib.Path: A Path object representing the full path to the labels file.
        
    Raises:
        FileNotFoundError: If the labels file is not found.
        RuntimeError: If there is an error loading the labels file.
    """
    try:
        labels_file = pathlib.Path(path) / f"{model_name}.json"
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file '{path}/{model_name}.json' not found.")
    except Exception as e:
        raise RuntimeError(f"Error loading labels '{path}/{model_name}.json': {e}")    
    return labels_file