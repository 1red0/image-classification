import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import sys

def load_class_names(labels_file):
    """Load class names from a text file."""
    with open(labels_file, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names

def preprocess_image(image_path, img_height, img_width):
    """Preprocess the image to the required size and format."""
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    return img_array

def classify_image(model, img_array, class_names):
    """Classify the class of the input image using the trained model."""
    classifications = model.predict(img_array)
    score = tf.nn.softmax(classifications[0])
    classified_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return classified_class, confidence

def main():
    if len(sys.argv) != 3:
        print("Usage: python classify.py <model_name> <image_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    image_path = sys.argv[2]
    labels_file = os.path.join('labels', model_name + '.txt')

    model = load_model(os.path.join('models', model_name +'.keras'))

    img_height = 256
    img_width = 256

    class_names = load_class_names(labels_file)
    img_array = preprocess_image(image_path, img_height, img_width)
    classified_class, confidence = classify_image(model, img_array, class_names)

    print(f"This image most likely belongs to {classified_class} with a {confidence:.2f} percent confidence.")

if __name__ == '__main__':
    main()
