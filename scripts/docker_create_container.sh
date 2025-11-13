# launch in project root
docker stop manipurl 2> /dev/null || true
docker rm manipurl 2> /dev/null || true
docker run -d --gpus all \
    --name manipurl \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -v "$(pwd)":/manipurl \
    manipurl \
    tail -f /dev/null