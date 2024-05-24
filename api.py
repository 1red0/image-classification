import pathlib
import logging
import aiofiles
import os
import tensorflow as tf
import numpy as np
import uvicorn
import asyncio
from werkzeug.utils import secure_filename
from typing import List, Tuple, AsyncGenerator
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import asynccontextmanager

# Initialize logging
logging.basicConfig(level=logging.INFO)

model_lock = asyncio.Lock()
model = None

def load_model(model_name: str) -> tf.keras.Model:
    """
    Load a TensorFlow Keras model from a specified file path.

    Args:
        model_name (str): The name of the model to be loaded (without extension).

    Returns:
        tf.keras.Model: The loaded Keras model object.

    Raises:
        FileNotFoundError: If the model file is not found.
        RuntimeError: If an error occurs during model loading.
    """
    model_path = os.path.join('models', model_name + '.keras')
    logging.info(f"Attempting to load model from {model_path}")
    
    if not os.path.exists(model_path):
        error_msg = f"Model file not found: {model_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        return tf.keras.models.load_model(model_path)
    except (IOError, tf.errors.NotFoundError) as e:
        error_msg = f"Failed to load model: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    An asynchronous context manager that initializes a global model variable by loading the first model alphabetically

    Args:
        app (FastAPI): An instance of FastAPI web application framework.

    Yields:
        None

    """
    global model
    models_dir = pathlib.Path('models')
    models = sorted([model.stem for model in models_dir.glob("*.keras")])
    
    if models:
        model_name = models[0]
        model = load_model(model_name)
    else:
        logging.warning("No models found in the models directory.")
    
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class SetModelRequest(BaseModel):
    new_model_name: str

@app.get("/models", response_model=List[str])
async def get_models():
    """
    Fetches and lists the names of machine learning models stored in the 'models' directory with a '.keras' extension.

    Returns:
        List[str]: A list of model names.
    """
    try:
        models_dir = pathlib.Path('models')
        models = sorted([model.stem for model in models_dir.glob("*.keras")])
        return JSONResponse(content={"models": models})
    except Exception as e:
        logging.error(f"Failed to fetch models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")
    
@app.get("/num_labels")
async def get_num_labels(model_name: str):
    """
    Retrieves the number of labels from a text file corresponding to a given model name.

    Args:
        model_name (str): The name of the model whose labels file needs to be accessed.

    Returns:
        JSONResponse: Returns a JSON object containing the number of labels if successful.
        HTTPException: Returns an error response if the file does not exist or an exception occurs.
    """
    try:
        labels_file_path = os.path.join('labels', model_name + '.txt')
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'r') as labels_file:
                num_labels = sum(1 for _ in labels_file)
            return JSONResponse(content={"num_labels": num_labels})
        else:
            raise HTTPException(status_code=404, detail="Labels file not found.")
    except Exception as e:
        logging.error(f"Failed to fetch labels: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch labels: {e}")

img_height = 256
img_width = 256

async def load_class_names(labels_file: str) -> List[str]:
    """
    Asynchronously reads a text file containing class names, one per line, and returns them as a list of strings.

    Args:
        labels_file (str): Path to the text file containing class names.

    Returns:
        List[str]: List of class names extracted from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If there is a permission issue accessing the file.
        IOError: For any other exceptions while reading the file.
    """
    try:
        async with aiofiles.open(labels_file, 'r') as file:
            class_names = [line.strip() async for line in file]
        return class_names
    except (FileNotFoundError, PermissionError) as e:
        error_msg = f"{e.__class__.__name__} occurred while trying to access {labels_file}: {e}"
        logging.error(error_msg)
        raise type(e)(error_msg)
    except Exception as e:
        error_msg = f"Failed to load class names from {labels_file}: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)

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

def classify_image(model: tf.keras.Model, img_array: tf.Tensor, class_names: List[str], top_k: int = 3) -> Tuple[List[str], List[float]]:
    """
    Predicts the class of an image using a pre-trained TensorFlow Keras model.
    
    Args:
        model (tf.keras.Model): A TensorFlow Keras model used for image classification.
        img_array (tf.Tensor): Preprocessed image tensor ready for model input.
        class_names (List[str]): List of strings representing class names corresponding to model output.
        top_k (int): Number of top predictions to return.
        
    Returns:
        Tuple[List[str], List[float]]: A tuple containing top predicted classes and their confidence scores.
    """
    try:
        classifications = model.predict(img_array)
        scores = tf.nn.softmax(classifications[0])
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        top_classes = [class_names[idx] for idx in top_indices]
        top_confidences = [float(f'{scores[idx] * 100:.2f}') for idx in top_indices]
        
        return top_classes, top_confidences
    except (tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError) as e:
        logging.error(f"Failed to classify image: {e}")
        raise RuntimeError(f"Failed to classify image: {e}")

@app.post("/classify")
async def classify(file: UploadFile = File(...), top_k: int = 3) -> JSONResponse:
    """
    Process an uploaded image file, validate it, classify it using a pre-trained model, and return predictions.

    Args:
        file (UploadFile): Uploaded image file.
        top_k (int): Number of top predictions to return (default is 3).

    Returns:
        JSONResponse: JSON response containing classifications and confidence levels.
    """
    async with model_lock:
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
            logging.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model")
async def set_model(request: SetModelRequest):
    """
    Update the global model used for inference by loading a new model specified by the user.

    Args:
        request (SetModelRequest): An instance of SetModelRequest containing the new_model_name attribute.

    Returns:
        JSONResponse: A JSON response indicating the successful change of the model, or an HTTP exception detailing the error encountered.
    """
    global model_name, model
    async with model_lock:
        try:
            new_model = load_model(request.new_model_name)
            model_name = request.new_model_name
            model = new_model
            message = f"Model changed to {request.new_model_name}"
            logging.info(message)
            return JSONResponse(content={"message": message})
        except (FileNotFoundError, RuntimeError) as e:
            error_type = "File not found" if isinstance(e, FileNotFoundError) else "Runtime error"
            logging.error(f"{error_type}: {e}")
            status_code = 404 if isinstance(e, FileNotFoundError) else 500
            raise HTTPException(status_code=status_code, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_index_html():
    """
    Serves the index.html file from the static directory with HTTP caching, handling file not found and server errors.
    
    Returns:
        HTMLResponse: HTML content of index.html or appropriate error response.
    """
    try:
        async with aiofiles.open(pathlib.Path("static") / "index.html", mode="r") as file:
            content = await file.read()
        response = HTMLResponse(content=content)
        response.headers["Cache-Control"] = "max-age=3600"
        return response
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: '{e}'")

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
