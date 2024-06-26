FROM continuumio/miniconda3

RUN mkdir dissertation
COPY . ./dissertation
WORKDIR /dissertation

# RUN conda env update --file environment.yml --name base
RUN conda env create --name diss_env --file environment.yml

RUN echo "conda activate diss_env" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]
