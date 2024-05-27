from typing import List
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

def tf_extra_configs() -> None:
    """
    Configures TensorFlow to use mixed precision, combining 32-bit and 16-bit floating point types for efficiency.
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def tf_custom_callbacks(model_name: str) -> List:
    """
    Create a list of TensorFlow callbacks commonly used for monitoring and adjusting the training process of a model.

    Args:
        model_name (str): The name of the model used for saving model checkpoints.

    Returns:
        list: List of TensorFlow callback instances configured with specified monitoring and adjustment settings.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, verbose=1, mode='auto', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, mode='auto', patience=3, min_delta=1e-6, cooldown=100, min_lr=0),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'models/{model_name}_checkpoint.keras', save_best_only=False)
    ]
    return callbacks