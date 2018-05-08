# A Dockerfile that sets up a full Gym install with test dependencies
FROM ubuntu:16.04

ARG MUJOCO_KEY
RUN echo $MUJOCO_KEY

#Install python3.6 on ubuntu 16.04
RUN apt-get -y update && apt-get install -y keyboard-configuration
#    apt-get install -y software-properties-common && \
#    add-apt-repository -y ppa:jonathonf/python-3.6 && \
#    apt-get -y update && \
#    apt-get -y install python3.6 python3.6-distutils

RUN apt-get install -y \ 
    python-setuptools \
    python-pip \
    libpq-dev \
    zlib1g-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    python-opengl \
    python-numpy \
    python-pyglet \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    libosmesa6-dev \
    patchelf \
    wget \
    unzip \
    git \
    xpra \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install tox

RUN mkdir /root/.mujoco && \
    cd /root/.mujoco && \
    echo $MUJOCO_KEY | base64 --decode > mjkey.txt && \
    wget https://www.roboti.us/download/mjpro150_linux.zip && \
    unzip mjpro150_linux.zip 

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
    
COPY . /usr/local/gym/
RUN cd /usr/local/gym && \
    tox --notest

RUN apt-get -y update && apt-get -y install vim && apt-get clean && \
    echo "set expandtab number shiftwidth=4 tabstop=4" > /root/.vimrc

RUN cp /usr/local/gym/xorg.conf /etc/X11/

# Finally, clean cached code (including dot files) and upload our actual code!
# RUN mv .tox /tmp/.tox && rm -rf .??* * && mv /tmp/.tox .tox
# COPY . /usr/local/gym/

WORKDIR /usr/local/gym/
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
