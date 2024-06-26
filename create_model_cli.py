import argparse
import logging
import os
import pathlib

from utils.config_utils import set_logging_level, set_memory, tf_custom_callbacks, tf_extra_configs
from services.creation_services import build_model, load_datasets, save_class_names

def main():
    """
    Orchestrates the process of training a deep learning model for image classification using TensorFlow.
    Handles user inputs for configuration, sets up the GPU memory, prepares the dataset, builds and trains the model,
    and saves the trained model and class labels.
    """
    try:
        set_logging_level(logging.INFO)

        set_memory()
        tf_extra_configs();

        parser = argparse.ArgumentParser(description='Train a deep learning model for image classification using TensorFlow.')
        parser.add_argument('--model_name', type=str, required=True, help='Name of the model (required)')
        parser.add_argument('--data_dir', type=str, required=False, default='data', help='Path to the dataset directory (default=data)')
        parser.add_argument('--epochs', type=int, required=False, default=15, help='Number of epochs to train the model (default=15)')
        parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size (default=32)')
        parser.add_argument('--img_height', type=int, required=False, default=256, help='Processing height of the images (default=256)')
        parser.add_argument('--img_width', type=int, required=False, default=256, help='Processing width of the images (default=256)')
        parser.add_argument('--validation_split', type=float, required=False, default=2e-1, help='Validation split (default=2e-1)')

        args = parser.parse_args()

        model_name = args.model_name
        data_dir = args.data_dir
        epochs = args.epochs
        batch_size = args.batch_size
        img_height = args.img_height
        img_width = args.img_width
        validation_split = args.validation_split

        data_dir = pathlib.Path(data_dir).with_suffix('')

        os.makedirs('labels', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        datasets = load_datasets(data_dir, img_height, img_width, batch_size, validation_split)
        labels = datasets['labels']
        train_ds = datasets['train_ds']
        val_ds = datasets['val_ds']

        save_labels_to = pathlib.Path('labels') / f"{model_name}.json"
        save_class_names(labels, save_labels_to)
        
        save_labels_checkpoint_to = pathlib.Path('labels') / f"{model_name}_checkpoint.json"
        save_class_names(labels, save_labels_checkpoint_to)

        num_classes = len(labels)

        model = build_model(num_classes=num_classes, 
                            img_height=img_height, 
                            img_width=img_width, 
                            regularization_rate=1e-3, 
                            min_dropout_rate=3e-1,
                            max_dropout_rate=5e-1, 
                            learning_rate=1e-4)
        
        callbacks = tf_custom_callbacks(model_name);

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        save_models_to = pathlib.Path('models') / f"{model_name}.keras"
        model.save(save_models_to)

    except KeyboardInterrupt:
        logging.info("Model creation interrupted. Exiting gracefully.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()