FROM python:3.8-buster

RUN apt-get update
RUN apt-get install libgl1 ffmpeg libsm6 libxext6 python3-dev build-essential -y

RUN mkdir app
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt