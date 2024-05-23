import argparse
import os
import pathlib
import tensorflow as tf
from PIL import Image

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

def convert_images_to_rgba(directory: str) -> None:
    """
    Convert images with transparency to RGBA format.

    Args:
    - directory: A string representing the path to the directory containing images to be processed.
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            if is_image_file(filepath):
                try:
                    with Image.open(filepath) as img:
                        if img.mode in ("P", "RGBA"):
                            img = img.convert("RGBA")
                            img.save(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

def is_image_file(filepath: str) -> bool:
    """Check if a file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.isfile(filepath) and os.path.splitext(filepath)[1].lower() in image_extensions

def load_datasets(data_dir: str, img_height: int, img_width: int, batch_size: int, validation_split: float = 0.2) -> dict:
    """
    Load training and validation datasets from directory.

    Args:
    - data_dir: The directory path where the image data is stored.
    - img_height: The height to which each image will be resized.
    - img_width: The width to which each image will be resized.
    - batch_size: The number of images to process in each batch.
    - validation_split: The fraction of data to reserve for validation.

    Returns:
    - dict: A dictionary with keys 'labels', 'train_ds', and 'val_ds'.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise ValueError(f"Invalid data directory: {data_dir}")
    convert_images_to_rgba(data_dir)
    
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=30,
        shear_range=0.2,
        horizontal_flip=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        brightness_range=(0.8, 1.2),
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_ds = data_gen.flow_from_directory(
        data_dir,        
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
        shuffle=True,
    )
    
    val_ds = data_gen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
        shuffle=True,
    )

    labels = list(train_ds.class_indices.keys())

    return {'labels': labels, 'train_ds': train_ds, 'val_ds': val_ds}

def save_class_names(labels: list, filename: str) -> None:
    """
    Save class names to a file.

    Args:
    - labels (list): A list of strings, each representing a class name.
    - filename (str): The path to the file where the class names will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for class_name in labels:
                f.write(f"{class_name}\n")
    except IOError as e:
        raise IOError(f"Error saving class names to {filename}: {e}")

def build_model(num_classes: int, img_height: int, img_width: int, regularization_rate: float, min_dropout_rate: float, max_dropout_rate: float, learning_rate: float) -> tf.keras.Model:
    """
    Build and compile a custom CNN model with improvements.

    Args:
        num_classes (int): The number of unique classes in the dataset.
        img_height (int): The height of the images that the model will process.
        img_width (int): The width of the images that the model will process.
        regularization_rate (float): Regularization rate for Dense layers.
        min_dropout_rate (float): Minimum dropout rate for Dropout layers.
        max_dropout_rate (float): Maximum dropout rate for Dropout layers.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model ready for training on image data.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(img_height, img_width, 3)),

        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),        
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(min_dropout_rate),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
        tf.keras.layers.Dropout(max_dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,  
        beta_1=0.9,             
        beta_2=0.999,           
        epsilon=1e-07           
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """
    Orchestrates the process of training a deep learning model for image classification using TensorFlow.
    Handles user inputs for configuration, sets up the GPU memory, prepares the dataset, builds and trains the model,
    and saves the trained model and class labels.
    """
    try:
        parser = argparse.ArgumentParser(description='Train a deep learning model for image classification using TensorFlow.')
        parser.add_argument('--model_name', type=str, required=True, help='Name of the model (required)')
        parser.add_argument('--data_dir', type=str, default='data', help='Path to the dataset directory (default=data)')
        parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model (default=15)')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default=32)')
        parser.add_argument('--img_height', type=int, default=256, help='Processing height of the images (default=256)')
        parser.add_argument('--img_width', type=int, default=256, help='Processing width of the images (default=256)')

        args = parser.parse_args()

        model_name = args.model_name
        data_dir = args.data_dir
        epochs = args.epochs
        batch_size = args.batch_size
        img_height = args.img_height
        img_width = args.img_width

        data_dir = pathlib.Path(data_dir).with_suffix('')

        os.makedirs('labels', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        set_memory()

        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        datasets = load_datasets(data_dir, img_height, img_width, batch_size)
        labels = datasets['labels']
        train_ds = datasets['train_ds']
        val_ds = datasets['val_ds']

        save_labels_to = pathlib.Path('labels') / f"{model_name}.txt"
        save_class_names(labels, save_labels_to)

        num_classes = len(labels)

        model = build_model(num_classes=num_classes, 
                            img_height=img_height, 
                            img_width=img_width, 
                            regularization_rate=0.001, 
                            min_dropout_rate=0.3,
                            max_dropout_rate=0.5, 
                            learning_rate=0.0001)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001, verbose=1, mode='auto', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, mode='auto', patience=3, min_delta=0.000001, cooldown=100, min_lr=0),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'models/{model_name}_best_accuracy.keras', monitor='accuracy', save_best_only=True)
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        save_models_to = pathlib.Path('models') / f"{model_name}.keras"
        model.save(save_models_to)

    except KeyboardInterrupt:
        print("\nModel creation interrupted. Exiting gracefully.")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except IOError as e:
        print(f"IO error occurred: {e}")
    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()