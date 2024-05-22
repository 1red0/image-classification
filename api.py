import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_name = os.environ['USE_MODEL']

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

def classify_image(model, img_array, class_names, top_k=3):
    """Classify the class of the input image using the trained model."""
    classifications = model.predict(img_array)
    scores = tf.nn.softmax(classifications[0])
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    top_classes = [class_names[idx] for idx in top_indices]
    top_confidences = [100 * scores[idx] for idx in top_indices]
    
    return top_classes, top_confidences


@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    top_k = int(request.args.get('top_k', 3))
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        class_names = load_class_names(os.path.join('labels', model_name + '.txt'))
        img_array = preprocess_image(file_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k=top_k)
        top_confidences = [float(f'{conf:.2f}') for conf in top_confidences]
        return jsonify({
            'classifications': [{'class': c, 'confidence': conf} for c, conf in zip(top_classes, top_confidences)]
        })


if __name__ == '__main__': 
    if not os.path.exists('uploads'):
        os.makedirs('uploads')    
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
