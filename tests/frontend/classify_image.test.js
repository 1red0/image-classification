import { classify_image } from "../../static/scripts";

describe('classify_image', () => {
    // No image file selected
    it('should alert when no image file is selected', async () => {
        // Given
        document.body.innerHTML = `
            <input type="file" id="file" />
            <div id="results"></div>
            <select id="num-labels">
                <option value="5">5</option>
            </select>
        `;

        const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});

        // When
        await classify_image();

        // Then
        expect(alertMock).toHaveBeenCalledWith('Please select an image');
    });

});
