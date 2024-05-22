# Image-classification

A image classification model written in Python using Tensorflow

## Instructions

### Python

Install dependencies `pip install -r requirements.txt`

To create a new model: `python create_model.py`

Inputs for `create_model.py`:

```text
Enter the path to the dataset directory (default: 'data'):
Enter the model name (default: 'model'):
Enter the number of epochs to train the model (default: 15):
Enter the batch size (default: 32):
Enter the processing height of the image (default: 256):
Enter the processing width of the image (default: 256):
```

To use the model `python classify.py`

Inputs for `classify.py`:

```text
Enter the model name (default: model):
Enter the path to the image:
Enter the number of classes to display (default: 3):
Enter the processing height of the image (default: 256):
Enter the processing width of the image (default: 256):
```

### API

#### Starting the API with Python

To launch the API `python api.py`

#### Starting the API with Docker

First build the image: `docker build -t image-classifier-app .`

Running the container using docker run: `docker run -p 5000:5000 image-classifier-app`

Running the container using docker-compose: `docker-compose up -d`

#### Using the API

Parameters:

```text
top_k=n -> the number of n classes to show

file=@path_to_image -> the image path
```

Using curl from the CLI: `curl -X POST -F file=@path_to_image http://localhost:5000/classify?top_k=n`

Using curl and jq from the CLI: `curl -X POST -F file=@path_to_image http://localhost:5000/classify?top_k=n | jq .`

Output for the API response (Example for top_k=3):

```json
{
    "classifications": [
        {
            "class": "class1",
            "confidence": 00.00
        },
        {
            "class": "class2",
            "confidence": 00.00
        },
        {
            "class": "class3",
            "confidence": 00.00
        }
    ]
}
```
