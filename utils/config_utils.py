import logging
import tensorflow as tf

def set_memory() -> None:
    """
    Set memory growth for GPUs to avoid OOM errors.

    This function configures TensorFlow to enable memory growth on all available GPUs.
    This prevents TensorFlow from allocating all the GPU memory at once, which helps in avoiding out-of-memory (OOM) errors during model training or inference.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")

def set_logging_level(level: int) -> None:
    """
    Configures the logging level of the Python logging module.

    Args:
        level: A logging level from the `logging` module (e.g., `logging.INFO`, `logging.DEBUG`).

    Returns:
        None

    Example:
        set_logging_level(logging.DEBUG)  # Set the logging level to DEBUG
    """
    logging.basicConfig(level=level)