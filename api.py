import os
import tensorflow as tf
import numpy as np
import uvicorn
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_name = os.environ['USE_MODEL']
except KeyError:
    raise EnvironmentError("Environment variable 'USE_MODEL' not set.")

try:
    model = tf.keras.models.load_model(os.path.join('models', model_name + '.keras'))
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

img_height = 256
img_width = 256

def load_class_names(labels_file):
    """Load class names from a text file."""
    try:
        with open(labels_file, 'r') as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names
    except Exception as e:
        raise IOError(f"Failed to load class names from {labels_file}: {e}")

def preprocess_image(image_path, img_height, img_width):
    """Preprocess the image to the required size and format."""
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image {image_path}: {e}")

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
        raise RuntimeError(f"Failed to classify image: {e}")

@app.post("/classify")
async def classify(file: UploadFile = File(...), top_k: int = 3):
    try:
        file_path = os.path.join('uploads', file.filename)
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        class_names = load_class_names(os.path.join('labels', model_name + '.txt'))
        img_array = preprocess_image(file_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k=top_k)
        top_confidences = [float(f'{conf:.2f}') for conf in top_confidences]
        
        return JSONResponse(content={
            'classifications': [{'class': c, 'confidence': conf} for c, conf in zip(top_classes, top_confidences)]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}")
