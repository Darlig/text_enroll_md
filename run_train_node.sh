#!/bin/bash

stage=0
world_size=6
rank="0,1"
config=config/EfficientNetb0_example.yaml
step=0
seed=4011
host="127.0.0.1"
port=22121
GPU="0,1"
pwd=`pwd`

. ./local/parse_options.sh

GPU=(`echo $GPU | awk -F',' '{for(x=1;x<=NF;x++) printf($x" ")}'`)
rank=(`echo $rank | awk -F',' '{for(x=1;x<=NF;x++) printf($x" ")}'`)

if [ $stage -le 1 ];then
    for id in ${!rank[@]}; do {
          this_rank=${rank[$id]}
	  this_gpu=${GPU[$id]}
          echo CUDA_VISIBLE_DEVICES=$this_gpu python3 -B train_node.py \
              --config $config \
             --world_size $world_size \
             --rank $this_rank \
             --gpu $this_gpu \
             --step $step \
             --seed $seed \
             --host $host \
	     --port $port
          CUDA_VISIBLE_DEVICES=$this_gpu python3 -B train_node.py \
              --config $config \
             --world_size $world_size \
             --rank $this_rank \
             --gpu $this_gpu \
             --step $step \
             --seed $seed \
             --host $host \
	     --port $port
    } &
    sleep 5
    done
fi
