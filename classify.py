import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

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

def classify_image(model, img_array, class_names, top_k=3):
    """Classify the class of the input image using the trained model."""
    classifications = model.predict(img_array)
    scores = tf.nn.softmax(classifications[0])
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    top_classes = [class_names[idx] for idx in top_indices]
    top_confidences = [100 * scores[idx] for idx in top_indices]
    
    return top_classes, top_confidences

def main():
    model_name = input("Enter the model name (default: model): ") or 'model'
    image_path = input("Enter the path to the image: ")
    top_k = int(input("Enter the number of classes to display (default: 3): ") or 3)
    img_height = int(input("Enter the processing height of the image (default: 256): ") or 256)
    img_width = int(input("Enter the processing width of the image (default: 256): ") or 256)
    
    labels_file = os.path.join('labels', model_name + '.txt')

    model = load_model(os.path.join('models', model_name + '.keras'))

    class_names = load_class_names(labels_file)
    img_array = preprocess_image(image_path, img_height, img_width)
    top_classes, top_confidences = classify_image(model, img_array, class_names, top_k=top_k)

    print("Classifications:")
    for label, confidence in zip(top_classes, top_confidences):
        print(f"- {label}: {confidence:.2f}% confidence")

if __name__ == '__main__':
    main()
