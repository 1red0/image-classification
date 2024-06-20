import pytest
import json

from services.classify_services import load_class_names

class TestLoadClassNames:

    # Successfully loads class names from a valid JSON file
    def test_loads_class_names_successfully(self, mocker):
        # Given
        labels_file = 'valid_labels.json'
        expected_class_names = ["class1", "class2", "class3"]
        mock_open = mocker.mock_open(read_data=json.dumps(expected_class_names))
        mocker.patch('builtins.open', mock_open)
    
        # When
        result = load_class_names(labels_file)
    
        # Then
        assert result == expected_class_names

    # Raises FileNotFoundError when the file does not exist
    def test_raises_file_not_found_error(self, mocker):
        # Given
        labels_file = 'non_existent_file.json'
        mocker.patch('builtins.open', side_effect=FileNotFoundError)
    
        # When / Then
        with pytest.raises(FileNotFoundError):
            load_class_names(labels_file)

    # Raises RuntimeError when JSON file is empty
    def test_raises_runtime_error_on_empty_json(self, mocker):
        # Given
        labels_file = 'empty_labels.json'
        mock_open = mocker.mock_open(read_data='')
        mocker.patch('builtins.open', mock_open)
    
        # When / Then
        with pytest.raises(RuntimeError, match="Error decoding JSON"):
            load_class_names(labels_file)

    # Handles JSON file with nested structures
    def test_handles_nested_structures(self, mocker):
        # Given
        labels_file = 'nested_labels.json'
        expected_class_names = ["class1", "class2", {"nested_class": "class3"}]
        mock_open = mocker.mock_open(read_data=json.dumps(expected_class_names))
        mocker.patch('builtins.open', mock_open)

        # When
        result = load_class_names(labels_file)

        # Then
        assert result == expected_class_names

    # Handles JSON file with multiple class names correctly
    def test_handles_json_file_with_multiple_class_names_correctly(self, mocker):
        # Given
        labels_file = 'multiple_labels.json'
        expected_class_names = ["class1", "class2", "class3"]
        mock_open = mocker.mock_open(read_data=json.dumps(expected_class_names))
        mocker.patch('builtins.open', mock_open)

        # When
        result = load_class_names(labels_file)

        # Then
        assert result == expected_class_names        