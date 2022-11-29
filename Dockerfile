FROM python:3.8-buster

RUN apt-get update
RUN xargs -a packages.txt sudo apt-get install -y

RUN mkdir app
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt