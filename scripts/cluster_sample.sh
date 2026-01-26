#!/bin/bash

#SBATCH -p 3090,A6000,RTX4090,RTX6000ADA,A5000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH -o /home/minwoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date


num_client=30
client_frac=1.0
data_path=~/dplora/news/data/30/0.01
data_name=news
lora_r=16
local_r=4
num_rounds=50
client_epochs=5
model=FacebookAI/roberta-base
mode=ncttlora
projection_type=global_mag
learning_rate=5e-4

tid=10890

python -u simul-server.py \
    --num_client $num_client --data_path $data_path --data_name $data_name \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r --local_r $local_r --client_lr $learning_rate --alter_lr 5.\
    --projection_type $projection_type --client_frac $client_frac --tid $tid \
    > outputs/${tid}.log 2>&1