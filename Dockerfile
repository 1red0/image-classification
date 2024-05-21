# Use the official TensorFlow image as base
FROM python:3.10

COPY models models

COPY labels labels

COPY api.py app.py

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
