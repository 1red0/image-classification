import pytest
from utils.image_utils import preprocess_image
import tensorflow as tf

class TestPreprocessImage:

    # Successfully preprocess a valid image file
    def test_successfully_preprocess_valid_image(self, mocker):
        # Arrange
        image_path = "valid_image.jpg"
        img_height = 224
        img_width = 224
        mocker.patch("tensorflow.io.read_file", return_value=tf.constant(b"fake_image_data"))
        mocker.patch("tensorflow.image.decode_image", return_value=tf.random.uniform([100, 100, 3]))
        mocker.patch("tensorflow.image.resize", return_value=tf.random.uniform([img_height, img_width, 3]))
    
        # Act
        result = preprocess_image(image_path, img_height, img_width)
    
        # Assert
        assert isinstance(result, tf.Tensor)
        assert result.shape == (1, img_height, img_width, 3)
        assert tf.reduce_max(result) <= 1.0
        assert tf.reduce_min(result) >= 0.0

    # Image path does not exist
    def test_image_path_does_not_exist(self, mocker):
        # Arrange
        image_path = "non_existent_image.jpg"
        img_height = 224
        img_width = 224
        mocker.patch("tensorflow.io.read_file", side_effect=FileNotFoundError("File not found"))
    
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            preprocess_image(image_path, img_height, img_width)