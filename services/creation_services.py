import os
import tensorflow as tf

from utils.image_utils import convert_images_to_rgba

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
        beta_1=9e-1,             
        beta_2=999e-3,           
        epsilon=1e-07           
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_datasets(data_dir: str, img_height: int, img_width: int, batch_size: int, validation_split: float) -> dict:
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
        horizontal_flip=True,
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
