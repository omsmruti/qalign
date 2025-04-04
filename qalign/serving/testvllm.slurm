#!/bin/bash
#SBATCH --job-name=vllm_server    # Job name
#SBATCH --output=vllm_%j.log      # Output file (%j will be replaced with job ID)
#SBATCH --error=vllm_%j.err       # Error file
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --gres=gpu:2             # Request 1 GPU
#SBATCH --time=24:00:00          # Time limit (24 hours)
#SBATCH --mem=128G                # Memory limit per node
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=32   
#SBATCH --account=cse          # Adding account specification
#SBATCH --qos=ckpt-gpu             # Adding QoS specification

# Load any required modules (adjust based on your system)
#module load cuda/11.8
#module load python/3.9

# Activate virtual environment if you're using one
# source /path/to/your/venv/bin/activate
source /gscratch/ark/graf/miniconda3/etc/profile.d/conda.sh  # or ~/anaconda3/etc/profile.d/conda.sh
conda activate quest38

MODEL="allenai/Llama-3.1-Tulu-3-8B-SFT"
PARALLEL_SIZE=2
MAX_NEW_TOKENS=800
MAX_PROMPT_LENGTH=1200
# Run the serverhay
python -m qflow.serving.vllm_serve \
    --model=$MODEL \
    --port=8000 \
    --host="0.0.0.0" \
    --tensor_parallel_size=$PARALLEL_SIZE \
    --gpu_memory_utilization=0.95 \
    --max_new_tokens=$MAX_NEW_TOKENS \
    --max_prompt_length=$MAX_PROMPT_LENGTH