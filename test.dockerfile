# A Dockerfile that sets up a full Gym install
FROM ubuntu:14.04

ARG MUJOCO_KEY

RUN apt-get -y update 
RUN apt-get install -y \
    libav-tools 

RUN apt-get install -y \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools

RUN apt-get install -y \
    python-opengl

RUN apt-get install -y \
    xpra

RUN apt-get install -y \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    libboost-all-dev \
    wget \
    unzip \
    git \
    vim \
    python3-dev 

RUN apt-get install -y libsdl2-2.0-0 
RUN apt-get install -y libsdl2-dev
RUN apt-get install -y python-setuptools

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && easy_install pip

RUN pip install tox

RUN echo "set number expandtab tabstop=4 shiftwidth=4" > /root/.vimrc
RUN mkdir /root/.mujoco && echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
RUN cd /root/.mujoco && wget https://www.roboti.us/download/mjpro131_linux.zip && unzip mjpro131_linux.zip

COPY . /usr/local/gym
RUN cd /usr/local/gym && tox --notest

WORKDIR /usr/local/gym
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
