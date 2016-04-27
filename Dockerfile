# A Dockerfile that sets up a full Gym install
FROM ubuntu:14.04

RUN apt-get update \
    && apt-get install -y xorg-dev \
    libgl1-mesa-dev \
    xvfb \
    libxinerama1 \
    libxcursor1 \
    libglu1-mesa \
    libav-tools \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && easy_install pip

WORKDIR /usr/local/gym
RUN mkdir gym && touch gym/__init__.py
COPY ./gym/version.py ./gym
COPY ./requirements.txt .
COPY ./setup.py .
RUN pip install -r requirements.txt

# Finally, upload our actual code!
COPY . /usr/local/gym

WORKDIR /root
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
