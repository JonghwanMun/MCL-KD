#! /bin/bash

options=$1
m_type=$2
dataset=$3
debug=$6

if [ ${debug} -eq 1 ] 
then
	CUDA_VISIBLE_DEVICES=$4 python -m src.experiment.tune_params \
		--config_path src/experiment/options/${dataset}/${m_type}/${options}.yml \
		--model_type ${m_type} \
		--dataset ${dataset} \
		--num_workers $5 \
		--debug_mode \
		#--interactive
else
	CUDA_VISIBLE_DEVICES=$4 python -m src.experiment.tune_params \
		--config_path src/experiment/options/${dataset}/${m_type}/${options}.yml \
		--model_type ${m_type} \
		--dataset ${dataset} \
		--num_workers $5 \
		#--debug_mode \
		#--interactive
fi
