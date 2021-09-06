# A Dockerfile based on CentOS 8 that sets up a full Gym install with test dependencies
FROM centos:8
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION:-36}

# Install pre-requisites
RUN dnf clean all
RUN dnf install -y epel-release
RUN dnf install -y --enablerepo=powertools wget unzip cmake gcc patchelf \
		python$PYTHON_VERSION python$PYTHON_VERSION-devel python3-pyopengl freeglut-devel \
		mesa-libOSMesa-devel mesa-libOSMesa mesa-libGL-devel glfw-devel xorg-x11-server-Xvfb
RUN rm -rf /var/cache/dnf

# Download and install mujoco
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    mv /root/.mujoco/mujoco200_linux/ /root/.mujoco/mujoco200/ && \
    echo DUMMY_KEY > /root/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/

# Set Python 3 as default interpreter
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN python --version

# Install required Python modules
RUN pip install pyglet && pip install ffmpeg && pip install -U 'mujoco-py<2.1,>=2.0' && pip install -r test_requirements.txt

# Execution
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
