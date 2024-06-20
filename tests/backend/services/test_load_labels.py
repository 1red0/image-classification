import pytest
import pathlib

from services.classify_services import load_labels

class TestLoadLabels:

    # Correct model name and path returns valid Path object
    def test_correct_model_name_and_path_returns_valid_path_object(self, mocker):
        # Given
        model_name = "test_model"
        path = "/valid/path"
        expected_path = pathlib.Path(path) / f"{model_name}.json"
        mocker.patch("pathlib.Path.exists", return_value=True)
    
        # When
        result = load_labels(model_name, path)
    
        # Then
        assert result == expected_path