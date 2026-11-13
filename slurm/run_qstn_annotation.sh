#!/bin/bash
#SBATCH --job-name=GABMAnnotation
#SBATCH --output=slurm_logs/GABMAnnot_%A/out.log
#SBATCH --error=slurm_logs/GABMAnnot_%A/err.log
##SBATCH --time=5:00:00
#SBATCH --nodes=1
##SBATCH --array=0-2   # Adjust based on number of parameter sets
#SBATCH --gres=gpu:1
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


### Instruct Models
models=(
    "zai-org/GLM-4.7-Flash"
)

# Optionally only annotate the responses of one source model: sbatch run_qstn_annotation.sh <source_model>
SOURCE_FLAG=""
if [ $# -ge 1 ]; then
    SOURCE_FLAG="--source_model $1"
fi

# Loop through each model and run the script
for model_id in "${models[@]}"; do
    python ../vote_prediction/qstn_response_annotation.py --model_id "$model_id" $SOURCE_FLAG $DEBUG_FLAG
done