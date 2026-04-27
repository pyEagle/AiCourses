#! /bin/bash

cd /usr/rfzn/Agent/agent
/home/rfzn/anaconda3/bin/conda activate agent
nohup python vx007.py >/dev/null 2>&1 &

