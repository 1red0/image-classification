import pytest
import json
import os

from services.creation_services import save_class_names

class TestSaveClassNames:

    # saves a list of class names to a specified file
    def test_saves_class_names_to_file(self, mocker):
        # Given
        labels = ["class1", "class2", "class3"]
        filename = "test_output/classes.json"
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_makedirs = mocker.patch("os.makedirs")

        # When
        save_class_names(labels, filename)

        # Then
        mock_makedirs.assert_called_once_with(os.path.dirname(filename), exist_ok=True)
        mock_open.assert_called_once_with(filename, 'w', encoding='utf-8')
        
        handle = mock_open()
        expected_content = json.dumps(labels, ensure_ascii=False, indent=4)

        # Collect all write calls
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        
        # Verify the written content
        assert written_data == expected_content

    # handles an empty list of class names
    def test_handles_empty_list_of_class_names(self, mocker):
        # Given
        labels = []
        filename = "test_output/empty_classes.json"
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_makedirs = mocker.patch("os.makedirs")

        # When
        save_class_names(labels, filename)

        # Then
        mock_makedirs.assert_called_once_with(os.path.dirname(filename), exist_ok=True)
        mock_open.assert_called_once_with(filename, 'w', encoding='utf-8')
        handle = mock_open()
        handle.write.assert_called_once_with(json.dumps(labels, ensure_ascii=False, indent=4))

    # raises an IOError if the file cannot be written
    def test_raises_io_error_if_file_cannot_be_written(self, mocker):
        # Given
        labels = ["class1", "class2", "class3"]
        filename = "test_output/classes.json"
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_open.side_effect = IOError("Mocked IOError")
        mock_makedirs = mocker.patch("os.makedirs")

        # When, Then
        with pytest.raises(IOError):
            save_class_names(labels, filename)
        mock_makedirs.assert_called_once_with(os.path.dirname(filename), exist_ok=True)
        mock_open.assert_called_once_with(filename, 'w', encoding='utf-8')

