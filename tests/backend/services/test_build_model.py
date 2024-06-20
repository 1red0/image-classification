
import pytest

import tensorflow as tf

from services.creation_services import build_model

class TestBuildModel:

    # model is created with valid parameters and compiles successfully
    def test_model_creation_and_compilation(self, mocker):
        # Given
        num_classes = 10
        img_height = 64
        img_width = 64
        regularization_rate = 0.01
        min_dropout_rate = 0.2
        max_dropout_rate = 0.5
        learning_rate = 0.001

        # When
        model = build_model(num_classes, img_height, img_width, regularization_rate, min_dropout_rate, max_dropout_rate, learning_rate)

        # Then
        assert isinstance(model, tf.keras.Model)
        assert model.optimizer.learning_rate == learning_rate

    # num_classes is set to 1
    def test_num_classes_is_one(self, mocker):
        # Given
        num_classes = 1
        img_height = 64
        img_width = 64
        regularization_rate = 0.01
        min_dropout_rate = 0.2
        max_dropout_rate = 0.5
        learning_rate = 0.001

        # When
        model = build_model(num_classes, img_height, img_width, regularization_rate, min_dropout_rate, max_dropout_rate, learning_rate)

        # Then
        assert model.output_shape == (None, num_classes)

    # img_height or img_width is set to 0 or negative values
    def test_invalid_image_dimensions(self, mocker):
        # Given
        num_classes = 10
        img_height = -1
        img_width = 64
        regularization_rate = 0.01
        min_dropout_rate = 0.2
        max_dropout_rate = 0.5
        learning_rate = 0.001

        # When / Then
        with pytest.raises(ValueError):
            build_model(num_classes, img_height, img_width, regularization_rate, min_dropout_rate, max_dropout_rate, learning_rate)

    # dropout layers are correctly applied with min_dropout_rate and max_dropout_rate
    def test_dropout_layers_application(self, mocker):
        # Given
        num_classes = 10
        img_height = 64
        img_width = 64
        regularization_rate = 0.01
        min_dropout_rate = 0.2
        max_dropout_rate = 0.5
        learning_rate = 0.001

        # When
        model = build_model(num_classes, img_height, img_width, regularization_rate, min_dropout_rate, max_dropout_rate, learning_rate)

        # Then
        assert any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)
        assert all(layer.rate >= min_dropout_rate and layer.rate <= max_dropout_rate for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout))
