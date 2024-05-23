import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import warnings

def load_class_names(labels_file):
    """Load class names from a text file."""
    try:
        with open(labels_file, 'r') as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file '{labels_file}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error loading class names from '{labels_file}': {e}")

def preprocess_image(image_path, img_height, img_width):
    """Preprocess the image to the required size and format."""
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image '{image_path}': {e}")

def classify_image(model, img_array, class_names, top_k=3):
    """Classify the class of the input image using the trained model."""
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
    warnings.filterwarnings("ignore")
    try:
        model_name = input("Enter the model name (default: model): ") or 'model'
        image_path = input("Enter the path to the image: ")
        top_k = int(input("Enter the number of classes to display (default: 3): ") or 3)
        img_height = int(input("Enter the processing height of the image (default: 256): ") or 256)
        img_width = int(input("Enter the processing width of the image (default: 256): ") or 256)

        labels_file = os.path.join('labels', model_name + '.txt')

        try:
            model = load_model(os.path.join('models', model_name + '.keras'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file 'models/{model_name}.keras' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading model 'models/{model_name}.keras': {e}")

        class_names = load_class_names(labels_file)
        img_array = preprocess_image(image_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k=top_k)

        print("Classifications:")
        for label, confidence in zip(top_classes, top_confidences):
            print(f"- {label}: {confidence:.2f}% confidence")
    except KeyboardInterrupt:
        print("\nClassification interrupted. Exiting gracefully.")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
