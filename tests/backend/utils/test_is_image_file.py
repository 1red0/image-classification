from utils.image_utils import is_image_file

class TestIsImageFile:

    # correctly identifies a .jpg file as an image
    def test_identifies_jpg_as_image(self, mocker):
        # Given
        filepath = "test_image.jpg"
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("os.path.splitext", return_value=("test_image", ".jpg"))
    
        # When
        result = is_image_file(filepath)
    
        # Then
        assert result is True

    # correctly identifies a file with uppercase extension as an image (e.g., .JPG)
    def test_identifies_uppercase_extension_as_image(self, mocker):
        # Given
        filepath = "test_image.JPG"
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("os.path.splitext", return_value=("test_image", ".JPG"))
    
        # When
        result = is_image_file(filepath)
    
        # Then
        assert result is True