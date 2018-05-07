# A Dockerfile that sets up a full Gym install with test dependencies
FROM ubuntu:16.04

RUN apt-get update -y && apt-get install -y keyboard-configuration
RUN apt-get install -y libav-tools \
    python-setuptools \
    python-pip \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    python-opengl \
    python-pyglet \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    libosmesa6 \
    patchelf \
    wget \
    unzip \
    git \
    xpra \
    libav-tools  \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install tox

COPY . /usr/local/gym/
RUN cd /usr/local/gym && \
    tox --notest

# Finally, clean cached code (including dot files) and upload our actual code!
# RUN mv .tox /tmp/.tox && rm -rf .??* * && mv /tmp/.tox .tox
# COPY . /usr/local/gym/

WORKDIR /usr/local/gym/
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
