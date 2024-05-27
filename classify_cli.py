import argparse
import logging

from services.classify_services import classify_image, load_class_names, load_labels, load_model, preprocess_image

def main():
    """
    Orchestrates the process of loading a machine learning model, preprocessing an image, classifying the image using the model,
    and displaying the classification results.
    """
    try:
        parser = argparse.ArgumentParser(description='Image classification script')
        parser.add_argument('--model_name', type=str, required=True, help='Name of the model (required)')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the image (required)')
        parser.add_argument('--top_k', type=int, default=3, help='Number of classes to display (default=3)')
        parser.add_argument('--img_height', type=int, default=256, help='Processing height of the image (default=256)')
        parser.add_argument('--img_width', type=int, default=256, help='Processing width of the image (default=256)')

        args = parser.parse_args()

        model_name = args.model_name
        image_path = args.image_path
        top_k = args.top_k
        img_height = args.img_height
        img_width = args.img_width

        model = load_model(model_name, 'models')
        labels_file = load_labels(model_name, 'labels')        

        class_names = load_class_names(labels_file)
        img_array = preprocess_image(image_path, img_height, img_width)
        top_classes, top_confidences = classify_image(model, img_array, class_names, top_k)

        print("Classifications:")
        for label, confidence in zip(top_classes, top_confidences):
            print(f"- {label}: {confidence:.2f}% confidence")
    except KeyboardInterrupt:
        logging.info("Classification interrupted. Exiting gracefully.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()