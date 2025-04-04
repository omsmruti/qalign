#!/bin/bash


RM=allenai/Llama-3.1-Tulu-3-8B-RM
DIR="outputs/"
query_args='{"variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'
python eval.py --base_dir $DIR --remote True --reward_model_path $RM --value_head False --batch_size 1024 --query_args $query_args

RM="/gscratch/ark/graf/LLaMA-Factory/saves/llama3/8b/full/reward/"
DIR="outputs-task/"
query_args='{"variant":"ancestral"}'
python eval.py --base_dir $DIR --remote True --reward_model_path $RM --value_head True --batch_size 1024 --query_args $query_args
