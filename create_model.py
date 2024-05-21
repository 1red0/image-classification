import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys

def set_memory_growth():
    """Set memory growth for GPUs to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def load_datasets(data_dir, img_height, img_width, batch_size):
    """Load training and validation datasets from directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    labels = train_ds.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return labels, train_ds, val_ds

def save_class_names(labels, filename):
    """Save class names to a file."""
    class_names = labels
    with open(filename, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    return class_names

def build_model(num_classes):
    """Build and compile the CNN model."""
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_model.py <model_name> <epochs>")
        sys.exit(1)    

    data_dir = pathlib.Path('data').with_suffix('')

    batch_size = 8
    img_height = 256
    img_width = 256

    model_name = sys.argv[1]
    epochs = int(sys.argv[2])
    
    set_memory_growth()
    
    labels, train_ds, val_ds = load_datasets(data_dir, img_height, img_width, batch_size)
    
    save_labels_to = os.path.join('labels', model_name + '.txt')

    class_names = save_class_names(labels, save_labels_to)
    num_classes = len(class_names)
    
    model = build_model(num_classes=num_classes)
    
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    model.save(os.path.join('models', model_name + '.keras'))

if __name__ == '__main__':
    main()
