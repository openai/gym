FROM gitpod/workspace-full-vnc

# Install dependencies
USER gitpod
RUN sudo apt-get -q update && \
    sudo apt-get install -yq python-opengl ffmpeg && \
    sudo rm -rf /var/lib/apt/lists/*
