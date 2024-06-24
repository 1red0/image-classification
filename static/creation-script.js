export async function create_new_model() {
    const modelName = document.getElementById("model-name").value;
    const epochs = parseInt(document.getElementById("model-epochs").value);
    const batchSize = parseInt(document.getElementById("model-batch_size").value);
    const imgHeight = parseInt(document.getElementById("model-img_height").value);
    const imgWidth = parseInt(document.getElementById("model-img_width").value);
    const validationSplit = parseFloat(
        document.getElementById("model-validation_split").value
    );

    const fileInput = document.getElementById('model-data');

    const new_model = {
        model_name: modelName,
        epochs: epochs,
        batch_size: batchSize,
        img_height: imgHeight,
        img_width: imgWidth,
        validation_split: validationSplit,
    };

    if (fileInput.files.length === 0) {
        alert('Please select an dataset');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('new_model', JSON.stringify(new_model));

    try {
        const response = await fetch(`/create-model`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

    } catch (error) {
        alert(`Model creation failed. ${error}`)
        return;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    // Add event listener for create-button click
    document.getElementById('create-button').addEventListener('click', async () => {
        await create_new_model();
    });

    document.getElementById('cancel-button').addEventListener('click', async () => {
        window.location.href = '/';
    });
});
