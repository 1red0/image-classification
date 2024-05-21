import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_name = os.environ['MODEL']

model = load_model(os.path.join('models/', model_name +'.keras'))

img_height = 256
img_width = 256

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
    """classify the class of the input image using the trained model."""
    classifications = model.predict(img_array)
    score = tf.nn.softmax(classifications[0])
    classified_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return classified_class, confidence

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        class_names = load_class_names(os.path.join('labels', model_name + '.txt'))
        img_array = preprocess_image(file_path, img_height, img_width)
        classified_class, confidence = classify_image(model, img_array, class_names)
        return jsonify({
            'classified_class': classified_class,
            'confidence': confidence
        })

if __name__ == '__main__': 
    if not os.path.exists('uploads'):
        os.makedirs('uploads')    
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
