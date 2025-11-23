# launch in project root
docker stop manipurl 2> /dev/null || true
docker rm manipurl 2> /dev/null || true
docker run -d --gpus all \
    --name manipurl2 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -v "$(pwd)":/manipurl \
    -p 5000:5000 \
    manipurl:v1 \
    tail -f /dev/null