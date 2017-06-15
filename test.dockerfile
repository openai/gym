# A Dockerfile that sets up a full Gym install
FROM quay.io/openai/gym:base

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
    software-properties-common \
    && apt-add-repository ppa:libccd-debs -y
    && apt-add-repository ppa:fcl-debs -y
    && apt-add-repository ppa:dartsim -y
    && apt-get update -q
    && apt-get install libdart6-all-dev -y
    && apt-get install swig -y
    && apt-get install swig python-pip python-qt4 python-qt4-dev python-qt4-gl -y
    && apt-get install python3-pip python3-pyqt4 python3-pyqt4.qtopengl -y
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && easy_install pip



WORKDIR /usr/local/gym/
RUN mkdir -p gym && touch gym/__init__.py
COPY ./gym/version.py ./gym/
COPY ./requirements.txt ./
COPY ./setup.py ./
COPY ./tox.ini ./

RUN pip install tox
# Install the relevant dependencies. Keep printing so Travis knows we're alive.
RUN ["bash", "-c", "( while true; do echo '.'; sleep 60; done ) & tox --notest"]

# Finally, clean cached code (including dot files) and upload our actual code!
RUN mv .tox /tmp/.tox && rm -rf .??* * && mv /tmp/.tox .tox
COPY . /usr/local/gym/

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
