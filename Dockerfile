FROM python:3.12-slim

COPY models /models

COPY labels /labels

COPY services /services

COPY utils /utils

COPY static /static

COPY server.py /server.py

COPY requirements.txt /requirements.txt

RUN pip --no-cache-dir install --upgrade pip==24.0 && \
    pip --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD ["python", "server.py"]
