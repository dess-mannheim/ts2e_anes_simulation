#!/bin/bash
#SBATCH --job-name=GABMEval
#SBATCH --output=slurm_logs/GABMEval_%A/out.log
#SBATCH --error=slurm_logs/GABMEval_%A/err.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --nodelist=compute2

# Executes the evaluation and regression notebooks in place (no GPU needed)

source /opt/anaconda3/etc/profile.d/conda.sh

# Kernel that runs the notebook with the gabm_vote_prediction environment (sklearn, zstandard, ...)
KERNEL_DIR=$(mktemp -d)
mkdir -p "$KERNEL_DIR/kernels/gabm_vote_prediction"
cat > "$KERNEL_DIR/kernels/gabm_vote_prediction/kernel.json" <<EOF
{"argv": ["$HOME/.conda/envs/gabm_vote_prediction/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "gabm_vote_prediction", "language": "python"}
EOF

cd ..

# Optional args: notebooks to run (in order), defaults to the evaluation followed by the regression analyses
if [ "$#" -gt 0 ]; then
    NOTEBOOKS=("$@")
else
    NOTEBOOKS=(
        eval_survey_responses_subgroup.ipynb
        regression.ipynb
        regression_llm_interactions.ipynb
        regression_individual.ipynb
    )
fi

# --allow-errors keeps going past failing exploratory cells; errors are stored in the notebook outputs
JUPYTER_PATH="$KERNEL_DIR" /opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace \
    --allow-errors \
    --ExecutePreprocessor.kernel_name=gabm_vote_prediction \
    --ExecutePreprocessor.timeout=-1 \
    "${NOTEBOOKS[@]}"
