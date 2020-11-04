# The image corresponding to this dosckerfile can be obtained by 
#docker pull us.gcr.io/broad-dsde-methods/pyro_matplotlib:0.0.7

FROM nvidia/cuda:10.2-base-ubuntu18.04

MAINTAINER Luca Dalessio <ldalessi@broadinstitue.org>
# System packages 
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
 && sudo apt-get install -y --no-install-recommends \
    git \
    bzip2 \
 && sudo rm -rf /var/lib/apt/lists/* \
 && curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /home/user/miniconda \
 && rm ~/miniconda.sh
# Put conda command on our path
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false


#----------- CONDA INSTALLS -------------

# CUDA 10.2-specific steps
RUN conda install -y pytorch==1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch \
 && conda clean -ya

RUN conda install -c anaconda psutil \
 && conda clean -ya

#----------- PIP INSTALLS --------------
RUN yes | pip install matplotlib==3.2.2
RUN yes | pip install plotly==4.5.0
RUN yes | pip install scikit-image==0.16.2
RUN yes | pip install neptune-client==0.4.120
RUN yes | pip install dill==0.3.2
RUN yes | pip install python-igraph==0.8.2
RUN yes | pip install leidenalg==0.8.1
RUN yes | pip install neptune-contrib
RUN yes | pip install --upgrade Pillow
RUN yes | pip install pyro-ppl==1.4.0
RUN yes | pip install ipython==7.13.0
                                             
