import pathlib
import logging
import aiofiles
import os
import tensorflow as tf
import numpy as np
import uvicorn
from werkzeug.utils import secure_filename
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def load_model(model_name: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(os.path.join('models', model_name + '.keras'))
    except (IOError, tf.errors.NotFoundError) as e:
        raise RuntimeError(f"Failed to load model: {e}")

try:
    model_name = os.environ['USE_MODEL']
except KeyError:
    raise EnvironmentError("Environment variable 'USE_MODEL' not set.")

model = load_model(model_name)

img_height = 256
img_width = 256

async def load_class_names(labels_file: str) -> List[str]:
    """
    Load class names from a text file.

    Args:
    - labels_file: A string specifying the path to the text file containing class names.

    Returns:
    - A list of class names, where each name is a string.
    """
    try:
        async with aiofiles.open(labels_file, 'r') as file:
            class_names = [line.strip() async for line in file]
        return class_names
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load class names from {labels_file}: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied while trying to access {labels_file}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load class names from {labels_file}: {e}")

def preprocess_image(image_path: str, img_height: int, img_width: int) -> tf.Tensor:
    """
    Preprocess the image to the required size and format.

    Args:
    - image_path: The file path of the image to be processed.
    - img_height: The target height of the image after resizing.
    - img_width: The target width of the image after resizing.

    Returns:
    A tensor representing the preprocessed image, ready to be fed into a neural network model.
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
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to preprocess image {image_path}: {e}")
    except tf.errors.InvalidArgumentError as e:
        raise tf.errors.InvalidArgumentError(f"Failed to preprocess image {image_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to preprocess image {image_path}: {e}")

def classify_image(model: tf.keras.Model, img_array: tf.Tensor, class_names: List[str], top_k: int = 3) -> Tuple[List[str], List[float]]:
    """
    Classify the class of the input image using the trained model.

    Args:
        model: A pre-trained TensorFlow model used for image classification.
        img_array: A preprocessed image tensor ready for model prediction.
        class_names: A list of class names corresponding to the model's output.
        top_k: An integer specifying the number of top predictions to return (default is 3).

    Returns:
        A tuple containing two lists:
        - top_classes: The names of the top predicted classes.
        - top_confidences: The confidence scores of these top classes, formatted as percentages.
    """
    try:
        classifications = model.predict(img_array)
        scores = tf.nn.softmax(classifications[0])
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        top_classes = [class_names[idx] for idx in top_indices]
        top_confidences = [float(f'{scores[idx] * 100:.2f}') for idx in top_indices]
        
        return top_classes, top_confidences
    except (tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError) as e:
        raise RuntimeError(f"Failed to classify image: {e}")

@app.post("/classify")
async def classify(file: UploadFile = File(...), top_k: int = 3):
    """
    Process an uploaded image file to predict its class using a pre-trained TensorFlow model.

    Args:
        file (UploadFile): Uploaded image file.
        top_k (int): Number of top predictions to return (default is 3).

    Returns:
        JSONResponse: JSON response containing the top class predictions and their confidence scores.
    """
    try:
        file_path = os.path.join('uploads', secure_filename(file.filename))
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Uploaded image file is too large.")
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        class_names = await load_class_names(os.path.join('labels', model_name + '.txt'))
        img_array = preprocess_image(file_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k=top_k)
        top_confidences = [float(f'{conf:.2f}') for conf in top_confidences]
        
        return JSONResponse(content={
            'classifications': [{'class': c, 'confidence': conf} for c, conf in zip(top_classes, top_confidences)]
        })
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    try:
        uploads_path = pathlib.Path('uploads')
        uploads_path.mkdir(parents=True, exist_ok=True)
        uvicorn.run(app, host="0.0.0.0", port=5200)
    except FileNotFoundError:
        logging.error('Failed to create uploads directory', exc_info=True)
    except OSError:
        logging.error('Failed to access file system', exc_info=True) 
    except Exception:
        logging.error('Failed to start server', exc_info=True) 
    except KeyboardInterrupt:
        logging.info("Server interrupted. Exiting gracefully.")

if __name__ == '__main__':
    start_server()
