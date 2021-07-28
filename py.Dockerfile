# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VER
FROM python:$PYTHON_VER
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig

# Download mujoco
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip

ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/
RUN pip install -e .[all]
RUN pip install -r test_requirements.txt

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
