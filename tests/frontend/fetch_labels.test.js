// Import fetch_labels function from your scripts.js
import { fetch_labels } from '../../static/scripts'

describe('fetch_labels', () => {

    // Fetches labels successfully when a valid model is provided
    it('should fetch and populate labels when a valid model is provided', async () => {
        // Given
        const model = 'validModel';
        const numLabelsSelect = document.createElement('select');
        numLabelsSelect.id = 'num-labels';
        document.body.appendChild(numLabelsSelect);
        const setModelButton = document.createElement('button');
        setModelButton.id = 'set-model-button';
        setModelButton.disabled = true;
        document.body.appendChild(setModelButton);

        global.fetch = jest.fn().mockResolvedValue({
            ok: true,
            json: jest.fn().mockResolvedValue({ num_labels: 5 })
        });

        // When
        await fetch_labels(model);

        // Then
        expect(fetch).toHaveBeenCalledWith(`/num_labels?model_name=${model}`);
        expect(numLabelsSelect.children.length).toBe(5);
    });

    // Handles scenarios where the response does not contain num_labels
    it('should handle scenarios where the response does not contain num_labels', async () => {
        // Given
        const model = 'noLabelsModel';
        const numLabelsSelect = document.createElement('select');
        numLabelsSelect.id = 'num-labels';
        document.body.appendChild(numLabelsSelect);
        const setModelButton = document.createElement('button');
        setModelButton.id = 'set-model-button';
        setModelButton.disabled = true;
        document.body.appendChild(setModelButton);
    
        global.fetch = jest.fn().mockResolvedValue({
            ok: true,
            json: jest.fn().mockResolvedValue({})
        });

        // When
        await fetch_labels(model);

        // Then
        expect(fetch).toHaveBeenCalledWith(`/num_labels?model_name=${model}`);
        expect(numLabelsSelect.children.length).toBe(0);
    });

    // Handles non-integer or negative num_labels in the response
    it('should handle non-integer or negative num_labels in the response', async () => {
        // Given
        const model = 'invalidModel';
        const numLabelsSelect = document.createElement('select');
        numLabelsSelect.id = 'num-labels';
        document.body.appendChild(numLabelsSelect);
        const setModelButton = document.createElement('button');
        setModelButton.id = 'set-model-button';
        setModelButton.disabled = true;
        document.body.appendChild(setModelButton);

        global.fetch = jest.fn().mockResolvedValue({
            ok: true,
            json: jest.fn().mockResolvedValue({ num_labels: -3 })
        });

        // When
        await fetch_labels(model);

        // Then
        expect(fetch).toHaveBeenCalledWith(`/num_labels?model_name=${model}`);
        expect(numLabelsSelect.children.length).toBe(0);
    });

    // Handles the scenario where the set-model-button is disabled
    it('should handle disabled set-model-button scenario', async () => {
        // Given
        const model = 'testModel';
        const numLabelsSelect = document.createElement('select');
        numLabelsSelect.id = 'num-labels';
        document.body.appendChild(numLabelsSelect);
        const setModelButton = document.createElement('button');
        setModelButton.id = 'set-model-button';
        setModelButton.disabled = true;
        document.body.appendChild(setModelButton);

        global.fetch = jest.fn().mockResolvedValue({
            ok: true,
            json: jest.fn().mockResolvedValue({ num_labels: 3 })
        });

        // When
        await fetch_labels(model);

        // Then
        expect(fetch).toHaveBeenCalledWith(`/num_labels?model_name=${model}`);
        expect(numLabelsSelect.children.length).toBe(0);
    });      
});