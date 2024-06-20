import pytest
from services.creation_services import load_datasets

class TestLoadDatasets:

    # Successfully loads datasets from a valid directory
    def test_load_datasets_success(self, mocker):
        # Given
        data_dir = "valid_data_dir"
        img_height = 224
        img_width = 224
        batch_size = 32
        validation_split = 0.2

        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('os.listdir', return_value=['class1', 'class2'])
        mock_convert_images_to_rgba = mocker.patch('services.creation_services.convert_images_to_rgba')

        # When
        result = load_datasets(data_dir, img_height, img_width, batch_size, validation_split)

        # Then
        mock_convert_images_to_rgba.assert_called_once_with(data_dir)
        assert 'labels' in result
        assert 'train_ds' in result
        assert 'val_ds' in result

    # Raises ValueError for non-existent data directory
    def test_load_datasets_non_existent_directory(self, mocker):
        # Given
        data_dir = "non_existent_data_dir"
        img_height = 224
        img_width = 224
        batch_size = 32
        validation_split = 0.2

        mocker.patch('os.path.exists', return_value=False)

        # When / Then
        with pytest.raises(ValueError, match=f"Invalid data directory: {data_dir}"):
            load_datasets(data_dir, img_height, img_width, batch_size, validation_split)

    # Raises ValueError for empty data directory
    def test_load_datasets_empty_directory(self, mocker):
        # Given
        data_dir = "empty_data_dir"
        img_height = 224
        img_width = 224
        batch_size = 32
        validation_split = 0.2

        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('os.listdir', return_value=[])

        # When / Then
        with pytest.raises(ValueError, match=f"Invalid data directory: {data_dir}"):
            load_datasets(data_dir, img_height, img_width, batch_size, validation_split)