# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig

# Download mujoco
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz &&\
    tar -xf mujoco210-linux-x86_64.tar.gz

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/

RUN if [ python:$PYTHON_VERSION = "python:3.6.15" ] ; then pip install .[box2d,classic_control,toy_text,other] pytest mock ; else pip install .[testing] ; fi

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
