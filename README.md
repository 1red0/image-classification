# Image-classification

A image classification model written in Python using Tensorflow

## Instructions

### Python

Install dependencies `pip install -r requirements.txt`

To create a new model: `python create_model.py`

Output for create model:

```text
Enter the path to the dataset directory (default: 'data'):
Enter the model name (default: 'model'):
Enter the number of epochs (default: 10):
Enter the batch size (default: 32):
Enter the image height (default: 256): 
Enter the image width (default: 256):
```

To use the model `python classify.py`

Output for classify:

```text
Enter the model name (default: model): 
Enter the path to the image:
Enter the processing height of the image (default: 256):
Enter the processing width of the image (default: 256):
```

### API

#### API-Python

To launch the API `python api.py`

#### API-Python-Docker

First build the image: `docker build -t image-classifier-app .`

Running the container from CLI: `docker run -p 5000:5000 image-classifier-app`

Running the container using docker-compose: `docker-compose up -d`

#### Use the API

`curl -X POST -F file=@image.* http://localhost:5000/classify`

Output for the API response:

```text
{"classified_class":"classified_class","confidence":confidence}
```
