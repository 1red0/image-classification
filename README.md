# image-classification

A image classification model written in Python using Tensorflow

## Variables set

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

## Run

### Python

Install dependencies ```pip install -r requirements.txt```

To create a new model: ```python create_model.py new_model 10``` where new_model is the name of the model and 10 are the training epochs

To use the model ```python classify.py model_name image.*```

To use the API ```python api.py``` then ```curl -X POST -F file=@image.* http://localhost:5000/classify```

### Docker

First build the image: ```docker build -t image-classifier-app .```

Run the container: ```docker run -p 5000:5000 image-classifier-app```
Using docker-compose: ```docker-compose up -d```

```curl -X POST -F file=@image.* http://localhost:5000/classify```
