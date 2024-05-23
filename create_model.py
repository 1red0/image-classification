import os
import pathlib
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.config.experimental import list_physical_devices, set_memory_growth


from PIL import Image
import warnings

def set_memory():
    """Set memory growth for GPUs to avoid OOM errors."""
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
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
    
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    labels = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)    

    return labels, train_ds, val_ds

def save_class_names(labels, filename):
    """Save class names to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for class_name in labels:
                f.write(f"{class_name}\n")
    except IOError as e:
        raise IOError(f"Error saving class names to {filename}: {e}")

def build_model(num_classes, img_height, img_width):
    """Build and compile a custom CNN model."""  

    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        
        layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),        
        layers.Rescaling(1./255),

        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),                 

        layers.Flatten(),

        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(
        learning_rate=0.00001,  
        beta_1=0.9,             
        beta_2=0.999,           
        epsilon=1e-07           
    )    

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

def main():
    warnings.filterwarnings("ignore")
    try:
        data_dir = input("Enter the path to the dataset directory (default: 'data'): ") or 'data'
        model_name = input("Enter the model name (default: 'model'): ") or 'model'
        epochs = int(input("Enter the number of epochs to train the model (default: 15): ") or 15)
        batch_size = int(input("Enter the batch size (default: 32): ") or 32)
        img_height = int(input("Enter the processing height of the image (default: 256): ") or 256)
        img_width = int(input("Enter the processing width of the image (default: 256): ") or 256)
        
        data_dir = pathlib.Path(data_dir).with_suffix('')

        set_memory()

        set_global_policy('mixed_float16')
        
        labels, train_ds, val_ds = load_datasets(data_dir, img_height, img_width, batch_size)
        
        os.makedirs('labels', exist_ok=True)
        save_labels_to = os.path.join('labels', model_name + '.txt')
        save_class_names(labels, save_labels_to)
        
        num_classes = len(labels)
        
        model = build_model(num_classes=num_classes, img_height=img_height, img_width=img_width)
            
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001)
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
