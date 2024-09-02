#!/bin/bash

stage=0
config=config/EfficientNetb0_example.yaml
step=0
seed=4011
port=22121
GPU="0,1"
pwd=`pwd`

. ./local/parse_options.sh

GPU=(`echo $GPU | awk -F',' '{for(x=1;x<=NF;x++) printf($x" ")}'`)
world_size=${#GPU[@]}

if [ $stage -le 1 ];then
    for id in `seq $world_size`;do {
          echo "Start $id"
          id=`expr $id - 1`
          gpu=${GPU["$id"]}
          CUDA_VISIBLE_DEVICES=$gpu python3 -B train.py \
              --config $config \
             --world_size $world_size \
             --rank $id \
             --gpu $gpu \
             --step $step \
             --seed $seed \
             --port $port
    } &
    sleep 5
    done
fi
