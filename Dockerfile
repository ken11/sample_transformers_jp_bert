FROM tensorflow/tensorflow:latest
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y && \
    apt install -y python3.8 python3.8-dev python3-pip && \
    apt clean && \
    python3.8 -m pip install --upgrade pip && python3.8 -m pip install transformers["ja"] tensorflow numpy sklearn

WORKDIR "/nlp"
