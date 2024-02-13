FROM postgres:15.2

# Sets default build args
ARG DB_USER=postgres
ARG DB_PASSWORD=password
ARG DB_NAME=mimic
ARG MIMIC_DATA_DIR=/var/data/mimic

# Installs system dependencies
RUN apt-get update -y && apt-get install -y git make

# Initialize processing files
RUN git clone https://github.com/MIT-LCP/mimic-code.git /tmp/mimic-code
ADD ./init.sh /tmp/mimic-code
WORKDIR /tmp/mimic-code
