FROM nvcr.io/nvidia/cuda:11.6.0-devel-ubuntu20.04
WORKDIR /test-repository-for-paper

# Conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH
RUN conda init bash

# Create a new Conda environment with the specified version of Python
RUN conda create -n integrating-sl-rl-for-ao python=3.8.13 -y

# Activate the environment by specifying it in the ENTRYPOINT
ENTRYPOINT ["conda", "run", "-n", "integrating-sl-rl-for-ao", "/bin/bash", "-c"]

# Other packages
RUN conda run -n integrating-sl-rl-for-ao conda install -c compass compass==5.2.1 -y
# Printing if compass is installed
RUN conda run -n integrating-sl-rl-for-ao conda list compass
RUN conda run -n integrating-sl-rl-for-ao pip install gym==0.26.2
RUN conda run -n integrating-sl-rl-for-ao pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117

ENV MAIN_DIR /test-repository-for-paper
ENV PYTHONPATH $MAIN_DIR:$PYTHONPATH
ENV SHESHA_ROOT $MAIN_DIR/shesha
ENV PYTHONPATH $SHESHA_ROOT:$PYTHONPATH
ENV PYTHONDONTWRITEBYTECODE 1

RUN conda run conda activate integrating-sl-rl-for-ao

# Ensures that the environment is activated
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "integrating-sl-rl-for-ao", "/bin/bash", "-c"]