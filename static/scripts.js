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
async function fetch_labels(model) {
    const selectedModel = model;
    const setModelButton = document.getElementById('set-model-button');

    if (setModelButton.disabled) {
        fetch(`/num_labels?model_name=${selectedModel}`)
            .then(response => response.json())
            .then(data => {
                const numLabelsSelect = document.getElementById('num-labels');
                numLabelsSelect.innerHTML = '';
                for (let i = 1; i <= data.num_labels; i++) {
                    const option = document.createElement('option');
                    option.value = i;
                    option.textContent = i;
                    numLabelsSelect.appendChild(option);
                }
            })
            .catch(error => alert(`Error fetching number of labels: ${error}`));
    }
}

/**
 * Asynchronous function to fetch models data from the server and populate a select element with the retrieved models.
 * Also adds an event listener to the select element to handle model selection and button disabling based on the selected model.
 * 
 * @returns {Promise<void>} A Promise that resolves once the models are fetched and the select element is populated.
 */
async function fetch_models() {
    try {
        const response = await fetch('/models');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        const modelSelect = document.getElementById('model');
        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });

        modelSelect.addEventListener('change', () => {
            const selectedModel = modelSelect.value;
            const setModelButton = document.getElementById('set-model-button');
            setModelButton.disabled = selectedModel === localStorage.getItem('selectedModel');
        });

        // Check if the model is already set and disable the button if true
        const selectedModel = localStorage.getItem('selectedModel');
        if (selectedModel) {
            modelSelect.value = selectedModel;
            // Manually trigger change event to fetch number of labels

            await fetch_labels(selectedModel);
        }

    } catch (error) {
        alert(`Error fetching the models: ${error}`)
        return;
    }
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

        // Enable the classify button
        const classifyButton = document.getElementById('classify-button');
        classifyButton.disabled = false;

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
async function displayImagePreview() {
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
async function classifyImage() {
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

    try {
        const response = await fetch(`/classify?top_k=${topK}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const classifications = data.classifications;

        resultsDiv.innerHTML = '<ul>' + classifications.map(c =>
            `<li>${c.class} - ${c.confidence}%</li>`
        ).join('') + '</ul>';
    } catch (error) {
        alert(`failed to classify. ${error}`)
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
    classifyButton.disabled = true;
    await set_model();
}

document.addEventListener("DOMContentLoaded", async () => {
    await fetch_models().then(() => {
        change_model_flow();
    })
});

document.getElementById('set-model-button').addEventListener('click', async () => {
    await clear_results();
    await change_model_flow();
});

document.getElementById('classify-button').addEventListener('click', async () => {
    await clear_results();
    await classifyImage();
});

document.getElementById('file').addEventListener('change', async () => {
    await clear_results();
    await clear_preview();
    await displayImagePreview();
});