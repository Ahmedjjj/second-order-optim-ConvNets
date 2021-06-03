FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG user
ARG usern


RUN mkdir -p /home
RUN mkdir -p /home/app

ENV PYTHONPATH="/home/app"
ENV PATH="$PATH:/home/.local/bin"

RUN useradd -d /home -s /bin/bash -u $user $usern
RUN chown -R $usern /home
USER $usern
WORKDIR /home/app

RUN pip --no-cache-dir --quiet install --upgrade matplotlib notebook pandas ipywidgets


