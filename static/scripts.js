window.addEventListener('beforeunload', () => {
    // Remove value from localStorage
    localStorage.removeItem('selectedModel');
});

/**
 * Asynchronous function to fetch labels data for a given model from the server and populate a select element with the retrieved labels.
 * 
 * @param {string} model - The model for which labels are to be fetched.
 * @returns {Promise<void>} A Promise that resolves once the labels are fetched and the select element is populated.
 */
export async function fetch_labels(model) {
    const numLabelsSelect = document.getElementById('num-labels');
    numLabelsSelect.innerHTML = '';

    try {
        const selectedModel = model;
        const setModelButton = document.getElementById('set-model-button');

        if (setModelButton.disabled) {
            const response = await fetch(`/num_labels?model_name=${selectedModel}`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            if (Number.isInteger(data.num_labels) && data.num_labels > 0) {
                for (let i = 1; i <= data.num_labels; i++) {
                    const option = document.createElement('option');
                    option.value = i;
                    option.textContent = i;
                    numLabelsSelect.appendChild(option);
                }
            }
        }
        // Enable the "Classify" button
        const classifyButton = document.getElementById('classify-button');
        classifyButton.disabled = false;
    } catch (error) {
        // Disable the "Classify" button
        const classifyButton = document.getElementById('classify-button');
        classifyButton.disabled = true;
        alert(`Error fetching number of labels: ${error}`);
    }
}

/**
 * Creates a debounced function that delays invoking the provided function until after 'delay' milliseconds have elapsed since the last time the debounced function was invoked.
 * This is useful for limiting the rate at which a function is executed, especially for expensive operations.
 * 
 * @param {Function} func - The function to debounce.
 * @param {number} delay - The number of milliseconds to delay.
 * @returns {Function} Returns the new debounced function.
 */
const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), delay);
    };
};

/**
 * Asynchronous function to fetch models data from the server and populate a select element with the retrieved models.
 * Also adds an event listener to the select element to handle model selection and button disabling based on the selected model.
 * 
 * @returns {Promise<void>} A Promise that resolves once the models are fetched and the select element is populated.
 */
export const fetch_models_data = async () => {
    try {
        const response = await fetch('/models');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        throw new Error(`Error fetching models data: ${error}`);
    }
};

/**
 * Asynchronously fetches models data and updates the model select dropdown with the retrieved models.
 * 
 * @returns {Promise<void>} A Promise that resolves once the models data is fetched and displayed.
 * @throws {Error} If there is an error fetching the models data or updating the model select dropdown.
 */
export async function fetch_models() {
    try {
        const data = await fetch_models_data();
        const modelSelect = document.getElementById('model');
        modelSelect.innerHTML = '';
        display_models(data.models, modelSelect);
        add_model_select_change_listener(modelSelect);
        await check_and_set_selected_model(modelSelect);
    } catch (error) {
        handle_fetch_models_error(error);
    }
}

/**
 * Displays the provided models in a select element.
 * 
 * @param {Array} models - An array of models to display.
 * @param {HTMLElement} modelSelect - The select element where the models will be displayed.
 * @returns {void}
 */
export function display_models(models, modelSelect) {
    if (models) {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    } else {
        alert("No models")
    }

}

/**
 * Adds a change event listener to the provided modelSelect element.
 * 
 * @param {HTMLElement} modelSelect - The HTML element representing the model selection dropdown.
 * 
 * This function adds a debounced change event listener to the modelSelect element. 
 * When the selection changes, it checks if the selected model is different from the one stored in localStorage.
 * If they are different, it enables the 'set-model-button'; otherwise, it disables the button.
 * The debounce function is used to limit the frequency of change events being triggered.
 */
function add_model_select_change_listener(modelSelect) {
    modelSelect.addEventListener('change', debounce(async () => {
        const selectedModel = modelSelect.value;
        const setModelButton = document.getElementById('set-model-button');
        setModelButton.disabled = selectedModel === localStorage.getItem('selectedModel');
    }, 300));
}

/**
 * Asynchronously checks if a selected model is stored in the localStorage and sets the value of the provided modelSelect element to the selected model if found. 
 * Then, it calls the fetch_labels function with the selected model as a parameter to fetch and display the labels for that model.
 * 
 * @param {HTMLElement} modelSelect - The HTML element representing the select input for models.
 * @returns {Promise<void>} - A Promise that resolves once the selected model is set and labels are fetched.
 */
async function check_and_set_selected_model(modelSelect) {
    const selectedModel = localStorage.getItem('selectedModel');
    if (selectedModel) {
        modelSelect.value = selectedModel;
        await fetch_labels(selectedModel);
    }
}

/**
 * Updates the error message element on the page to display an error message related to fetching models.
 * 
 * @param {Error} error - The error that occurred while fetching models data.
 * @returns {void}
 */
function handle_fetch_models_error(error) {
    alert(`Error fetching number of labels: ${error}`);
}

/**
 * Asynchronous function to set the selected model by sending a POST request to the server with the new model name.
 * Saves the selected model to localStorage and disables the "Set Model" button upon successful setting.
 * Displays an alert if an error occurs while setting the model.
 * 
 * @returns {Promise<void>} A Promise that resolves once the model is successfully set and saved to localStorage.
 */
async function set_model() {
    const modelSelect = document.getElementById('model');
    const modelName = modelSelect.value;

    try {
        const response = await fetch('/set_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ new_model_name: modelName }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Save the selected model to localStorage
        localStorage.setItem('selectedModel', modelName);

        // Disable the "Set Model" button
        const setModelButton = document.getElementById('set-model-button');
        setModelButton.disabled = true;

        // Enable the "Create New Model" button
        const createButton = document.getElementById('create-new-model-button');
        createButton.disabled = false;

        await fetch_labels(modelName);

    } catch (error) {
        alert(`Model selection failed. ${error}`)
        return;
    }
}

/**
 * Asynchronous function to clear the content of the image preview div by setting its innerHTML to an empty string.
 * 
 * @returns {Promise<void>} A Promise that resolves once the image preview div content is cleared.
 */
async function clear_preview() {
    const imagePreviewDiv = document.getElementById('image-preview');
    imagePreviewDiv.innerHTML = '';
}

/**
 * Asynchronous function to clear the content of the results div by setting its innerHTML to an empty string.
 * 
 * @returns {Promise<void>} A Promise that resolves once the results div content is cleared.
 */
async function clear_results() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
}

/**
 * Asynchronous function to display a preview of the selected image file in the specified image preview div.
 * 
 * @returns {Promise<void>} A Promise that resolves once the image preview is displayed.
 */
async function display_image_preview() {
    const fileInput = document.getElementById('file');
    const imagePreviewDiv = document.getElementById('image-preview');

    if (fileInput.files.length === 0) {
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreviewDiv.innerHTML = `<img src=${JSON.stringify(e.target.result)} alt="Image Preview">`;
    };
    reader.readAsDataURL(file);
}

/**
 * Asynchronous function to classify an image by sending it to the server for processing and displaying the results.
 * 
 * @returns {Promise<void>} A Promise that resolves once the image is classified and the results are displayed.
 */
export async function classify_image() {
    const fileInput = document.getElementById('file');
    const resultsDiv = document.getElementById('results');
    const numLabelsSelect = document.getElementById('num-labels');
    const topK = numLabelsSelect.value;

    if (fileInput.files.length === 0) {
        alert('Please select an image');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const classifyUrl = `/classify?top_k=${topK}`;

    try {
        const response = await fetch(classifyUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const classifications = data.classifications;

        // Display loading indicator
        resultsDiv.innerHTML = '<div class="loading-indicator">Classifying image...</div>';

        resultsDiv.innerHTML = '<ul>' + classifications.map(c =>
            `<li>${c.class} - ${c.confidence}%</li>`
        ).join('') + '</ul>';
    } catch (error) {
        alert(`Failed to classify. ${error}`);
        return;
    }
}

/**
 * Asynchronous function that controls the flow of changing the model.
 * Disables the classify button, fetches models, and sets the selected model.
 * 
 * @returns {Promise<void>} A Promise that resolves once the model is set.
 */
async function change_model_flow() {
    const classifyButton = document.getElementById('classify-button');
    const createButton = document.getElementById('create-new-model-button');

    classifyButton.disabled = true;
    createButton.disabled = true;
    await set_model();
}

// Wait for DOMContentLoaded event
document.addEventListener('DOMContentLoaded', async () => {

    await fetch_models().then(() => {
        change_model_flow();
    })

    // Add event listener for set-model-button click
    document.getElementById('set-model-button').addEventListener('click', async () => {
        await clear_results();
        await change_model_flow();
    });

    // Add event listener for create-new-model-button click
    document.getElementById('create-new-model-button').addEventListener('click', async () => {
        window.location.href = 'create-model';
    });

    // Add event listener for classify-button click
    document.getElementById('classify-button').addEventListener('click', async () => {
        await clear_results();
        await classify_image();
    });

    // Add event listener for file input change
    document.getElementById('file').addEventListener('change', async () => {
        await clear_results();
        await clear_preview();
        await display_image_preview();
    });

});
