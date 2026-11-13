#!/bin/bash
#SBATCH --job-name=GABMSurveySim
#SBATCH --output=slurm_logs/GABMSim_%A/out.log
#SBATCH --error=slurm_logs/GABMSim_%A/err.log
##SBATCH --time=5:00:00
#SBATCH --nodes=1
##SBATCH --array=0-2   # Adjust based on number of parameter sets
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --nodelist=compute2

source /opt/anaconda3/etc/profile.d/conda.sh

conda activate gabm_cuda13

export TORCH_CUDA_ARCH_LIST="12.0"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"

# Force Triton Backend
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP16=0

DEBUG_FLAG="${DEBUG_FLAG:-}" #"--debug" to enable, else "" (can also be set in the environment)


# The 6 models used in the case study (Section 5 of the paper).
# Comment out models you don't want to (re-)run, e.g. to split work across jobs.
models=(
    ### Qwen 3
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    ### Olmo
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3.1-32B-Instruct"
    ### Meta
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
)

# 5 random seeds used to assess simulation reliability (Section 5 of the paper)
seeds=(
    42
    43
    44
    45
    46
)

# Optionally override the lists above: sbatch run_qstn_vote_prediction.sh <model_id> <seed> [<seed> ...]
if [ $# -ge 2 ]; then
    models=("$1")
    seeds=("${@:2}")
fi

# Loop through each model and run the script
for model_id in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        python ../vote_prediction/qstn_vote_prediction.py --model_id "$model_id" --seed $seed $DEBUG_FLAG
    done
done