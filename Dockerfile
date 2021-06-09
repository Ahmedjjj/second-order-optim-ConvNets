FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG user
ARG usern

RUN mkdir -p /home
RUN mkdir -p /home/app

ENV PYTHONPATH="/home/app:/home/PyTorch-LBFGS/functions:/home/adahessian/image_classification:/home/apollo/optim"
ENV PATH="$PATH:/home/.local/bin"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y git

RUN useradd -d /home -s /bin/bash -u $user $usern
RUN chown -R $usern /home
USER $usern

RUN pip --no-cache-dir --quiet install --upgrade matplotlib notebook pandas ipywidgets

WORKDIR /home
RUN git clone https://github.com/hjmshi/PyTorch-LBFGS.git
RUN git clone https://github.com/amirgholami/adahessian.git
RUN git clone https://github.com/XuezheMax/apollo.git
WORKDIR /home/app
