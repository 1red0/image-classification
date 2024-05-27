import asyncio
import logging
import pathlib
from fastapi.staticfiles import StaticFiles
import uvicorn

from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from services.classify_services import load_model

model_lock = asyncio.Lock()
model = None

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
    models_dir = pathlib.Path('../models')
    models = sorted([model.stem for model in models_dir.glob("*.keras")])
    
    if models:
        model_name = models[0]
        model = load_model(model_name, '../models')
    else:
        logging.warning("No models found in the models directory.")
    
    yield

def create_app(path: str):
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=path), name="static")

    return app

def start_server(app: FastAPI):
    try:
        uploads_path = pathlib.Path('../uploads')
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