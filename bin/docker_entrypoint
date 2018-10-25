#!/bin/bash
# This script is the entrypoint for our Docker image.

set -ex

# Set up display; otherwise rendering will fail
Xvfb -screen 0 1024x768x24 &
export DISPLAY=:0

# Wait for the file to come up
display=0
file="/tmp/.X11-unix/X$display"
for i in $(seq 1 10); do
    if [ -e "$file" ]; then
	break
    fi

    echo "Waiting for $file to be created (try $i/10)"
    sleep "$i"
done
if ! [ -e "$file" ]; then
    echo "Timing out: $file was not created"
    exit 1
fi

exec "$@"
