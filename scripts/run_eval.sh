#!/bin/bash


directories=("outputs/" "outputs-task/")

echo "Comparing responses to GT for all datasets"

DIR="outputs/"

RM="lastnumber"
query_args='{"dataset":"openai/gsm8k"}'
python eval.py --base_dir "$DIR" --reward_model_path $RM --query_args $query_args

RM="lastmath"
query_args='{"dataset":"HuggingFaceH4/MATH-500"}'
python eval.py --base_dir "$DIR" --reward_model_path $RM  --query_args $query_args


RM="ifeval"
query_args='{"dataset":"google/IFEval"}'
python eval.py --base_dir "$DIR" --reward_model_path $RM  --query_args $query_args


RM="lastoption"

query_args='{"dataset":"edinburgh-dawg/mmlu-redux"}'
python eval.py --base_dir "$DIR" --reward_model_path $RM --query_args $query_args

query_args='{"dataset":"truthfulqa/truthful_qa"}'
python eval.py --base_dir "$DIR" --reward_model_path $RM  --query_args $query_args


DIR="outputs-task/"

RM="lastnumber"
query_args='{}'
python eval.py --base_dir "$DIR" --reward_model_path $RM --query_args $query_args