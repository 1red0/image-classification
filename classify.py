import argparse
import logging
import pathlib
from typing import List, Tuple
import tensorflow as tf
import numpy as np

def load_class_names(labels_file: str) -> List[str]:
    """
    Load class names from a text file.

    Args:
    - labels_file: A string representing the path to the text file containing class names.

    Returns:
    - A list of strings, where each string is a class name extracted from the file.
    """
    try:
        with open(labels_file, 'r') as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file '{labels_file}' not found.")
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
    except Exception as e:
        raise RuntimeError(f"Error classifying image: {e}")

def main():
    """
    Orchestrates the process of loading a machine learning model, preprocessing an image, classifying the image using the model,
    and displaying the classification results.
    """
    try:
        parser = argparse.ArgumentParser(description='Image classification script')
        parser.add_argument('--model_name', type=str, required=True, help='Name of the model (required)')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the image (required)')
        parser.add_argument('--top_k', type=int, default=3, help='Number of classes to display (default=3)')
        parser.add_argument('--img_height', type=int, default=256, help='Processing height of the image (default=256)')
        parser.add_argument('--img_width', type=int, default=256, help='Processing width of the image (default=256)')

        args = parser.parse_args()

        model_name = args.model_name
        image_path = args.image_path
        top_k = args.top_k
        img_height = args.img_height
        img_width = args.img_width

        labels_file = pathlib.Path('labels') / f"{model_name}.txt"

        try:
            model = tf.keras.models.load_model(pathlib.Path('models') / f"{model_name}.keras")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file 'models/{model_name}.keras' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading model 'models/{model_name}.keras': {e}")

        class_names = load_class_names(labels_file)
        img_array = preprocess_image(image_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k)

        print("Classifications:")
        for label, confidence in zip(top_classes, top_confidences):
            print(f"- {label}: {confidence:.2f}% confidence")
    except KeyboardInterrupt:
        logging.info("Classification interrupted. Exiting gracefully.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()