#!/bin/bash


echo "Running MV WMV and BON and for all datasets"


## general exps
DIR="outputs/"

RM="lastnumber"
query_args='{"dataset":"openai/gsm8k"}'
python pred.py --base_dir "$DIR" --extract $RM --strategy "voting" --query_args $query_args

RM="lastmath"

query_args='{"dataset":"HuggingFaceH4/MATH-500"}'
python pred.py --base_dir "$DIR" --extract $RM --strategy "voting" --query_args $query_args


RM="ifeval"

query_args='{"dataset":"google/IFEval"}'
python pred.py --base_dir "$DIR" --extract $RM --strategy "mbr" --query_args $query_args


RM="lastoption"

query_args='{"dataset":"edinburgh-dawg/mmlu-redux"}'
python pred.py --base_dir "$DIR" --extract $RM --strategy "voting" --query_args $query_args

query_args='{"dataset":"truthfulqa/truthful_qa"}'
python pred.py --base_dir "$DIR" --extract $RM --strategy "voting"  --query_args $query_args



## bon and WMV on general exps

RM="crm:allenai-Llama-3"

query_args='{"dataset":"google/IFEval","variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'

python pred.py --base_dir $DIR --strategy "bon" --extract "ifeval" --query_args $query_args --key $RM
python pred.py --base_dir $DIR --strategy "wmbr"  --extract "ifeval" --query_args $query_args  --key $RM

query_args='{"dataset":"HuggingFaceH4/MATH-500","variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'

python pred.py --base_dir $DIR  --strategy "bon" --extract "lastmath" --query_args $query_args  --key $RM
python pred.py --base_dir $DIR  --strategy "weighted-voting" --extract "lastmath" --query_args $query_args  --key $RM

query_args='{"dataset":"openai/gsm8k","variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'

python pred.py --base_dir $DIR  --strategy "bon"  --extract "lastnumber" --query_args $query_args  --key $RM
python pred.py --base_dir $DIR --strategy "weighted-voting"  --extract "lastnumber" --query_args $query_args  --key $RM

query_args='{"dataset":"truthfulqa/truthful_qa","variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'

python pred.py --base_dir $DIR  --strategy "bon" --extract "lastoption" --query_args $query_args  --key $RM
python pred.py --base_dir $DIR  --strategy "weighted-voting" --extract "lastoption" --query_args $query_args  --key $RM

query_args='{"dataset":"edinburgh-dawg/mmlu-redux","variant":"ancestral","model_path":"allenai/Llama-3.1-Tulu-3-8B-SFT"}'

python pred.py --base_dir $DIR  --strategy "bon"  --extract "lastoption" --query_args $query_args  --key $RM
python pred.py --base_dir $DIR  --strategy "weighted-voting"  --extract "lastoption" --query_args $query_args  --key $RM


### task specific exps
RM="vh:-gscratch-ark-graf-LLaMA-Factory-saves-llama3-8b-full-reward-"

DIR="outputs-task/"

query_args='{}'

python pred.py --base_dir "$DIR" --strategy "voting" --extract "lastnumber"  --query_args $query_args

query_args='{"variant":"ancestral"}'

python pred.py --base_dir $DIR --strategy "bon"  --extract "lastnumber" --query_args $query_args  --key $RM 
python pred.py --base_dir $DIR --strategy "weighted-voting"  --extract "lastnumber" --query_args $query_args  --key $RM
