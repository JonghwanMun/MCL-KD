#! /bin/bash

options=$1
m_type=$2
dataset=$3
CUDA_VISIBLE_DEVICES=$4 python -m src.calibration.eval \
	--exp ${options} \
	--model_type ${m_type} \
	--dataset ${dataset} \
	--start_epoch $5 \
	--end_epoch $6 \
	--epoch_stride $7 \
	--num_workers $8 \
    --temperature $9
