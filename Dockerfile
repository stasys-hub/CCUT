FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
COPY env.yml env.yml
RUN conda env create -f env.yml
#RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda --version
RUN conda list
RUN coverage run -m pytest -vv
