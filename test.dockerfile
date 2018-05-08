# A Dockerfile that sets up a full Gym install
FROM ubuntu:14.04

RUN apt-get update \
    && apt-get install -y libav-tools \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    xpra \
    libav-tools  \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && easy_install pip

RUN pip install tox
COPY . /usr/local/gym
RUN cd /usr/local/gym && tox --notest

WORKDIR /usr/local/gym
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
