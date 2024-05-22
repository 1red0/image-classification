# Image-classification

A image classification model written in Python using Tensorflow

## Run

### Python

Install dependencies `pip install -r requirements.txt`

To create a new model: `python create_model.py`

To use the model `python classify.py`

### API

#### API-Python

To launch the API `python api.py`

#### API-Python-Docker

First build the image: `docker build -t image-classifier-app .`

Run the container: `docker run -p 5000:5000 image-classifier-app`
Using docker-compose: `docker-compose up -d`

#### Use the API

`curl -X POST -F file=@image.* http://localhost:5000/classify`
