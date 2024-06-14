
// Import fetch_models_data function from your scripts.js
import { fetch_models_data } from '../../static/scripts'

describe('fetch_models_data', () => {

    // Successfully fetches models data from the server
    it('should return models data when the server responds with valid JSON', async () => {
        // Given
        const mockResponse = { models: ['model1', 'model2'] };
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            })
        );

        // When
        const data = await fetch_models_data();

        // Then
        expect(data).toEqual(mockResponse);
        expect(fetch).toHaveBeenCalledWith('/models');
    });

    // Server returns a non-JSON response
    it('should throw an error when the server responds with non-JSON data', async () => {
        // Given
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.reject(new Error('Invalid JSON')),
            })
        );

        // When / Then
        await expect(fetch_models_data()).rejects.toThrow('Error fetching models data: Error: Invalid JSON');
        expect(fetch).toHaveBeenCalledWith('/models');
    });

    // Network error occurs during fetch
    it('should throw an error when a network error occurs during fetch', async () => {
        // Given
        global.fetch = jest.fn(() => Promise.reject(new Error('Network Error')));

        // When / Then
        await expect(fetch_models_data()).rejects.toThrow('Error fetching models data: Error: Network Error');
        expect(fetch).toHaveBeenCalledWith('/models');
    });
});
