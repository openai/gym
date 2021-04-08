# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VER
FROM python:$PYTHON_VER
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig
RUN \ 
# Download mujoco
    mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip

ARG PYTHON_VER
ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
RUN pip install pytest pytest-forked lz4

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/
# install all extras for python 3.6 and 3.7, and skip mujoco add-ons for 3.8 and 3.9
# as mujoco 1.50 does not seem to work with 3.8 and 3.9
RUN bash -c "[[ $PYTHON_VER =~ 3\.[6-7]\.[0-9] ]] && pip install -e .[all] || pip install -e .[nomujoco]"

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["pytest","--forked"]
