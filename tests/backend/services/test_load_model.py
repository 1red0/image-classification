import pytest
import tensorflow as tf

from services.classify_services import load_model

class TestLoadModel:

    # Successfully loads a model when given a correct model name and path
    def test_loads_model_successfully(self, mocker):
        # Given
        model_name = "test_model"
        path = "test_path"
        mock_model = mocker.Mock(spec=tf.keras.Model)
        mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)
    
        # When
        result = load_model(model_name, path)
    
        # Then
        assert result == mock_model

    # Raises FileNotFoundError when the model file does not exist
    def test_raises_filenotfounderror_when_model_not_found(self, mocker):
        # Given
        model_name = "non_existent_model"
        path = "test_path"
        mocker.patch('tensorflow.keras.models.load_model', side_effect=FileNotFoundError)
    
        # When / Then
        with pytest.raises(FileNotFoundError):
            load_model(model_name, path)

    # Raises RuntimeError for corrupted or incompatible model files
    def test_raises_runtimeerror_for_corrupted_model(self, mocker):
        # Given
        model_name = "corrupted_model"
        path = "test_path"
        mocker.patch('tensorflow.keras.models.load_model', side_effect=RuntimeError("Corrupted file"))
    
        # When / Then
        with pytest.raises(RuntimeError):
            load_model(model_name, path)

    # Handles loading models from different directories correctly
    def test_loads_model_from_different_directories(self, mocker):
        # Given
        model_name = "test_model"
        path = "different_path"
        mock_model = mocker.Mock(spec=tf.keras.Model)
        mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)

        # When
        result = load_model(model_name, path)

        # Then
        assert result == mock_model

    # Handles very long model names or paths
    def test_handles_very_long_names_or_paths(self, mocker):
        # Given
        model_name = "a" * 1000
        path = "b" * 1000
        mock_model = mocker.Mock(spec=tf.keras.Model)
        mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)
    
        # When
        result = load_model(model_name, path)
    
        # Then
        assert result == mock_model

    # Handles paths with special characters or spaces
    def test_handles_special_characters_or_spaces(self, mocker):
        # Given
        model_name = "model with spaces"
        path = "path/with/special/characters"
        mock_model = mocker.Mock(spec=tf.keras.Model)
        mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)
    
        # When
        result = load_model(model_name, path)
    
        # Then
        assert result == mock_model