import logging
import os
from PIL import Image
import tensorflow as tf

img_height = 256
img_width = 256

def convert_images_to_rgba(directory: str) -> None:
    """
    Convert images with transparency to RGBA format.

    Args:
    - directory: A string representing the path to the directory containing images to be processed.
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            if is_image_file(filepath):
                try:
                    with Image.open(filepath) as img:
                        if img.mode in ("P", "RGBA"):
                            img = img.convert("RGBA")
                            img.save(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

def is_image_file(filepath: str) -> bool:
    """Check if a file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.isfile(filepath) and os.path.splitext(filepath)[1].lower() in image_extensions

def preprocess_image(image_path: str, img_height: int, img_width: int) -> tf.Tensor:
    """
    Reads an image from a specified path, decodes it, resizes it to given dimensions,
    normalizes its pixel values, and returns it as a TensorFlow tensor suitable for model input.

    Args:
        image_path (str): Path to the image file.
        img_height (int): Height to resize the image.
        img_width (int): Width to resize the image.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    if not isinstance(img_height, int) or not isinstance(img_width, int) or img_height <= 0 or img_width <= 0:
        raise ValueError("img_height and img_width must be positive integers.")
    
    try:
        logging.info(f"Preprocessing image: {image_path}")
        img = tf.io.read_file(image_path)
        logging.debug("Read image file successfully.")
        img = tf.image.decode_image(img, channels=3)
        logging.debug("Decoded image successfully.")
        img = tf.image.resize(img, [img_height, img_width])
        logging.debug("Resized image successfully.")
        img = tf.expand_dims(img, 0)
        logging.debug("Expanded dimensions of the image array.")
        img /= 255.0
        return img
    except (FileNotFoundError, tf.errors.InvalidArgumentError) as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        raise type(e)(f"Failed to preprocess image {image_path}: {e}")
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        raise ValueError(f"Failed to preprocess image {image_path}: {e}")