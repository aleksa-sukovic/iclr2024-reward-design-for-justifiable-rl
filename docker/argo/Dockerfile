FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG APP_HOME=/home/argo

# Sets environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Installs system dependencies
RUN apt-get update && apt-get install -y software-properties-common apt-transport-https
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update
RUN apt-get install -y \
    vim \
    git \
    software-properties-common \
    apt-transport-https \
    python3.9 \
    python3.9-venv
RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Sets up application
RUN mkdir -p ${APP_HOME}
COPY ./requirements.txt ${APP_HOME}
WORKDIR ${APP_HOME}

# Installs Python dependencies
RUN python -m ensurepip
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ./requirements.txt
