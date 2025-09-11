#!/bin/bash

stage=0
config=config/EfficientNetb0_example.yaml
step=0
seed=4011
port=22121
GPU="0,1"
resume_from=
pwd=`pwd`

. ./local/parse_options.sh

GPU=(`echo $GPU | awk -F',' '{for(x=1;x<=NF;x++) printf($x" ")}'`)
world_size=${#GPU[@]}
start_time=`date +%Y%m%d_%H%M%S`

if [ $stage -le 1 ];then
if [ -z $resume_from ]; then
    for id in `seq $world_size`;do {
          echo "Start $id"
          id=`expr $id - 1`
          gpu=${GPU["$id"]}
          CUDA_VISIBLE_DEVICES=$gpu python3 -B train_kw_init_fin_train_data_test.py \
              --config $config \
             --world_size $world_size \
             --rank $id \
             --gpu $gpu \
             --step $step \
             --seed $seed \
             --port $port \
             --start-time $start_time
    } &
    sleep 5
    done
    wait
else
    for id in `seq $world_size`;do {
          echo "Start $id"
          id=`expr $id - 1`
          gpu=${GPU["$id"]}
          CUDA_VISIBLE_DEVICES=$gpu python3 -B train_kw_init_fin_train_data_test.py \
              --config $config \
             --world_size $world_size \
             --rank $id \
             --gpu $gpu \
             --step $step \
             --seed $seed \
             --port $port \
             --start-time $start_time \
             --resume-from $resume_from
    } &
    sleep 5
    done
    wait
fi

fi
