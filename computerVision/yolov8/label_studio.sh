#! /bin/bash

if [ "$1" = "start" ];then
    /home/rfzn/anaconda3/bin/cond activate labelStudio
    export LOCAL_FILES_SERVING_ENABLED=true
    nohup label-studio start --host 0.0.0.0 --port 8080 --no-browser>/dev/null 2>&1 &
else
    nohup label-studio-ml start ./yolo_backend -p 9090 --host 0.0.0.0 >/dev/null 2>&1 &
    echo 'finish'
fi

