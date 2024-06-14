// Import fetch_models function from your scripts.js
import { display_models } from '../../static/scripts'

describe('display_models', () => {

    // Displays all provided models in the select element
    it('should display all provided models in the select element', () => {
        // Given
        const models = ['Model1', 'Model2', 'Model3'];
        document.body.innerHTML = '<select id="model"></select>';
        const modelSelect = document.getElementById('model');
    
        // When
        display_models(models, modelSelect);
    
        // Then
        expect(modelSelect.children.length).toBe(models.length);
        models.forEach((model, index) => {
            expect(modelSelect.children[index].value).toBe(model);
            expect(modelSelect.children[index].textContent).toBe(model);
        });
    });

    // Handles an empty models array gracefully
    it('should handle an empty models array gracefully', () => {
        // Given
        const models = [];
        document.body.innerHTML = '<select id="model"></select>';
        const modelSelect = document.getElementById('model');
    
        // When
        display_models(models, modelSelect);
    
        // Then
        expect(modelSelect.children.length).toBe(0);
    });

    // Handles a null or undefined models parameter
    it('should handle a null or undefined models parameter', () => {
        // Given
        const models = null;
        document.body.innerHTML = '<select id="model"></select>';
        const modelSelect = document.getElementById('model');
        window.alert = jest.fn();
    
        // When
        display_models(models, modelSelect);
    
        // Then
        expect(window.alert).toHaveBeenCalledWith("No models");
        expect(modelSelect.children.length).toBe(0);
    });
});
