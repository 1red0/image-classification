from typing import List
import logging
import pathlib
import uvicorn

from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.classify_services import load_model
from utils.global_vars_utils import model, model_lock

async def get_sorted_models(path: str) -> List[str]:
    """
    Asynchronously retrieves and sorts the names of Keras model files from a specified directory after validating the input path.

    Args:
        path (str): A string specifying the directory path where the Keras model files are stored.

    Returns:
        List[str]: A list of strings, where each string is the stem of a Keras model file sorted alphabetically.
    """
    async with model_lock:
        try:
            if not pathlib.Path(path).is_dir():
                raise ValueError("Invalid directory path provided.")

            models_dir = (pathlib.Path(__file__).parents[1]).joinpath(path)
            models = sorted([model.stem for model in models_dir.glob("*") if model.suffix == '.keras'])

            if not models:
                raise FileNotFoundError("No Keras model files found in the specified directory." )

            logging.info(f"Models sorted successfully: {models}")

            return models
        except (OSError) as e:
            logging.error(f'Failed to access directory: {path}. {e}', exc_info=True)
            return []

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
    
    models = await get_sorted_models('models')
    
    if (models != []):
        model_name = models[0]

        model = load_model(model_name, str((pathlib.Path(__file__).parents[1]).joinpath('models')))

        logging.info(f"Model selected: {model_name}")
    else:
        logging.warning('No models available - No model selected')
    
    yield

def create_app(path: str) -> FastAPI:
    """
    Initializes a FastAPI application with specific middleware for CORS handling and mounts a directory for serving static files.

    Args:
        path (str): A string specifying the directory path where static files are located.

    Returns:
        FastAPI: Configured FastAPI application instance ready to be run with a web server.
    """
    app = FastAPI(lifespan=lifespan)

    # Add CORS middleware to allow all origins, credentials, methods, and headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Mount the specified directory for serving static files under the URL path '/static'
    app.mount("/static", StaticFiles(directory=path), name="static")

    return app

def start_server(app: FastAPI):
    """
    Initializes a server for a FastAPI application, handling file system operations and server startup with error management.

    Args:
        app (FastAPI): The FastAPI application to be served.
    """
    try:
        uploads_path = (pathlib.Path(__file__).parents[1]).joinpath('uploads')
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