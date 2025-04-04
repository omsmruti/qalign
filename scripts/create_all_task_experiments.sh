#!/bin/bash


MODELPATH=meta-llama/Llama-3.1-8B-Instruct
DIR="outputs-task/"
RM_PATH="/gscratch/ark/graf/LLaMA-Factory/saves/llama3/8b/full/reward/"
TEMP=1.0
N=128
STEPS=4096
BATCH_SIZE=128

## ANCESTRAL 
DATASET="openai/gsm8k"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test" --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET

## QUEST
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type value --split "test" --num_chains 1 --batch_size $BATCH_SIZE --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 

DATASET="apple/GSM-Symbolic-p1"
## ANCESTRAL 
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test" --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET

## QUEST
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type value --split "test" --num_chains 1 --batch_size $BATCH_SIZE --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 
