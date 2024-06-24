import pathlib
import logging
import subprocess
import aiofiles
import os
import json
import zipfile

from pydantic import BaseModel, model_validator
from werkzeug.utils import secure_filename
from typing import List
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from services.classify_services import classify_image, load_class_names, load_model, preprocess_image
from utils.config_utils import set_logging_level
from utils.server_utils import create_app, start_server, model_lock, model
from utils.global_vars_utils import img_height, img_width

set_logging_level(logging.INFO)

app = create_app("./static")

class SetModelRequest(BaseModel):
    new_model_name: str

class CreateModelRequest(BaseModel):
    model_name: str
    epochs: int
    batch_size: int
    img_height: int
    img_width: int
    validation_split: float  

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value    

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
    Retrieves the number of labels from a JSON file corresponding to a given model name.

    Args:
        model_name (str): The name of the model whose labels file needs to be accessed.

    Returns:
        JSONResponse: Returns a JSON object containing the number of labels if successful.
        HTTPException: Returns an error response if the file does not exist or an exception occurs.
    """
    try:
        labels_file_path = os.path.join('./labels', model_name + '.json')
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'r', encoding='utf-8') as labels_file:
                labels = json.load(labels_file)
                num_labels = len(labels)
            return JSONResponse(content={"num_labels": num_labels})
        else:
            raise HTTPException(status_code=404, detail="Labels file not found.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{labels_file_path}'")
        raise HTTPException(status_code=500, detail="Error decoding JSON.")
    except Exception as e:
        logging.error(f"Failed to fetch labels: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch labels: {e}")

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
            file_path = os.path.join('uploads', 'images', secure_filename(file.filename))
            
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
            
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Uploaded image file is too large.")
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            class_names = load_class_names(os.path.join('labels', model_name + '.json'))
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
            new_model = load_model(request.new_model_name, 'models')
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

@app.get("/create-model", response_class=HTMLResponse)
async def serve_create_model_html():
    """
    Serves the create-model.html file from the static directory with HTTP caching, handling file not found and server errors.
    
    Returns:
        HTMLResponse: HTML content of create-model.html or appropriate error response.
    """
    try:
        async with aiofiles.open(pathlib.Path("static") / "create-model.html", mode="r") as file:
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

@app.post("/create-model")
async def create_model_endpoint(new_model: CreateModelRequest, file: UploadFile = File(...)):
    try:
        script_path = "./create_model_cli.py"  # Replace with actual path

        file_path = os.path.join('uploads', 'datasets', secure_filename(file.filename))

        # Read the file content once
        content = await file.read()

        if len(content) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="Uploaded training set is too large.")

        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
                logging.info("Archive uploaded successfully")                
        except Exception as e:
            logging.error(f"Error in writing the archive: {e}")
        finally:
            try:
                with zipfile.ZipFile(f"uploads/datasets/{file.filename}", "r") as zip_ref:
                    zip_ref.extractall('data/' + new_model.model_name + '_dataset')
                    logging.info("Archive extracted successfully")
            except Exception as e:
                logging.error(f"Error in extracting the archive: {e}")

        process = subprocess.Popen(
            ["python", script_path,
                f"--model_name {new_model.model_name}",
                f"--data_dir data/{new_model.model_name}_dataset",
                f"--epochs {new_model.epochs}",
                f"--batch_size {new_model.batch_size}",
                f"--img_height {new_model.img_height}",
                f"--img_width {new_model.img_width}",
                f"--validation_split {new_model.validation_split}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output, error = process.communicate(timeout=600)  # Set a timeout limit

        return output, error

    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: '{e}'")


if __name__ == '__main__':
    start_server(app)
