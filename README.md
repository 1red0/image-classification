# Image-classification

A image classification model written in Python using Tensorflow

## Run

### Python

Install dependencies `pip install -r requirements.txt`

To create a new model: `python create_model.py new_model 10` where new_model is the name of the model and 10 are the training epochs

To use the model `python classify.py model_name image.*`

### API

#### API-Python

To launch the API `python api.py`

#### API-Python-Docker

First build the image: `docker build -t image-classifier-app .`

Run the container: `docker run -p 5000:5000 image-classifier-app`
Using docker-compose: `docker-compose up -d`

#### Use the API

`curl -X POST -F file=@image.* http://localhost:5000/classify`
