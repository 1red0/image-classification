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
