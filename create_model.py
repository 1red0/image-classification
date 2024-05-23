import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

def set_memory_growth():
    """Set memory growth for GPUs to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

def is_image_file(filepath):
    """Check if a file is an image based on its extension."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif')
    return filepath.lower().endswith(valid_extensions)

def convert_images_to_rgba(directory):
    """Convert images with transparency to RGBA format."""
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

def load_datasets(data_dir, img_height, img_width, batch_size):
    """Load training and validation datasets from directory."""
    convert_images_to_rgba(data_dir)
    
    try:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified directory '{data_dir}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"Error loading datasets: {e}")

    labels = train_ds.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
    
    return labels, train_ds, val_ds

def save_class_names(labels, filename):
    """Save class names to a file."""
    class_names = list(labels)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
    except IOError as e:
        raise IOError(f"Error saving class names to {filename}: {e}")

    return class_names

def build_model(num_classes, img_height, img_width):
    """Build and compile the custom CNN model."""
    try:
        model = Sequential([
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    except Exception as e:
        raise RuntimeError(f"Error building model: {e}")
    
    return model

def main():
    try:
        data_dir = input("Enter the path to the dataset directory (default: 'data'): ") or 'data'
        model_name = input("Enter the model name (default: 'model'): ") or 'model'
        epochs = int(input("Enter the number of epochs to train the model (default: 15): ") or 15)
        batch_size = int(input("Enter the batch size (default: 32): ") or 32)
        img_height = int(input("Enter the processing height of the image (default: 256): ") or 256)
        img_width = int(input("Enter the processing width of the image (default: 256): ") or 256)
        
        data_dir = pathlib.Path(data_dir).with_suffix('')

        set_memory_growth()

        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        labels, train_ds, val_ds = load_datasets(data_dir, img_height, img_width, batch_size)
        
        os.makedirs('labels', exist_ok=True)
        save_labels_to = os.path.join('labels', model_name + '.txt')
        class_names = save_class_names(labels, save_labels_to)
        
        num_classes = len(class_names)
        
        model = build_model(num_classes=num_classes, img_height=img_height, img_width=img_width)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        os.makedirs('models', exist_ok=True)
        save_models_to = os.path.join('models', model_name + '.keras')
        model.save(save_models_to)

    except KeyboardInterrupt:
        print("\nModel creation interrupted. Exiting gracefully.")
    except FileNotFoundError as e:
        print(f"Directory error: {e}")
    except IOError as e:
        print(f"File I/O error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
