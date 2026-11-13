# ANES 2024 Vote Choice Simulation

Code for the empirical case study in **["Total Simulated Survey Error: An Evaluation Framework for
LLM-generated Survey Responses"](https://arxiv.org/abs/2609.10280)** (Section 5): simulating vote choice in the 2024 U.S. presidential election with LLM personas and comparing the results against the 2024 American National Election Studies (ANES) survey.

The case study runs a multiverse analysis over 5 design choices from the paper's TS2E framework —
**task articulation**, **persona construction**, **LLM selection**, **response generation/processing**,
and **adjustment** — giving 288 simulation configurations (144 of them run over 5 random seeds), for a
total of 3,651,120 simulated responses across 6 open-weight LLMs. Survey simulation uses the
[QSTN framework](https://github.com/dess-mannheim/QSTN/).

This repository  also contains the [TS2E checklist in a document format](https://github.com/dess-mannheim/ts2e_anes_simulation/blob/main/TS2E_checklist_template.docx).

## Repository layout

| Path | Purpose |
|---|---|
| `preprocess_ANES.ipynb` | Downloads/cleans the ANES 2024 data (pre-election variables only) into `data/2024_anes_pre-election_preprocessed.csv` (N=5,071) |
| `vote_prediction/templates.py` | Builds the 4 persona variants (demo vs. demo+attitudes, direct vs. interview format) |
| `vote_prediction/qstn_vote_prediction.py` | Runs the LLM vote-choice simulation for one model/seed across all task × persona × output-format combinations |
| `vote_prediction/qstn_response_annotation.py` | LLM-judge annotation of free-text responses (extracts Harris/Trump/Non-voter) |
| `slurm/run_qstn_vote_prediction.sh`, `slurm/run_qstn_annotation.sh` | Slurm batch scripts for the two steps above |
| `slurm/run_eval.sh` | Slurm batch script that executes the evaluation and regression notebooks in place |
| `slurm/submit_pre_rerun.py` | Submits the full pipeline (simulation, annotation, evaluation) as Slurm dependency chains |
| `eval_survey_responses_subgroup.ipynb` | Computes weighted F1 / TVD per configuration and per demographic subgroup; produces the reliability plot (Fig. 4) |
| `regression.ipynb` | OLS regressions of F1/TVD on the 5 design choices; Table 1 and the overall/gender/ideology lollipop plots (Figs. 5 & 8) |
| `regression_llm_interactions.ipynb` | Same regressions with LLM × design-choice interaction terms (Figs. 9 & 10) |
| `regression_individual.ipynb` | Individual-level accuracy and vote-share-by-ideology comparison vs. ANES ground truth (Fig. 7) |
| `results/pre_election/` | Outputs of the current pre-election design: simulations, annotations, evaluated results and figures |
| `results/pre_election/annotated_GLM-4.7-Flash_<model>.zst` | **Precomputed**: all simulated responses with parsed/judged vote choice, one file per LLM |
| `results/pre_election/evaluated_results_subgroup_nonadjusted.csv` | **Precomputed**: aggregated F1/TVD per configuration, backing Table 1 and all regressions |

The precomputed files let you skip straight to the evaluation and regression analysis (steps 4-5 below)
without re-running the GPU simulation and annotation. Results of an earlier version of the case study are
described [at the bottom](#earlier-results-mixed-pre--and-post-election-design).

## Setup

```bash
pip install qstn[vllm]  # includes vllm, torch, etc. — needed only for simulation/annotation (steps 1-2)
pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels stargazer regex tqdm zstandard  # analysis notebooks
```

The annotated responses (`results/*/annotated_GLM-4.7-Flash*.zst`) are tracked via Git LFS:
```bash
git lfs install && git lfs pull
```

## Reproducing the case study

### 1. Get the ANES data
Download the 2024 ANES Time Series Study CSV from
https://electionstudies.org/data-center/2024-time-series-study/, place it in `data/`, and run
`preprocess_ANES.ipynb` top to bottom. This writes `data/2024_anes_pre-election_preprocessed.csv`.

### 2. Simulate vote choice (GPU, optional — see precomputed file above)
For each of the 6 models (LLaMA-3.1-8B, LLaMA-3.3-70B, OLMo-3-7B, OLMo-3.1-32B, Qwen3-VL-8B,
Qwen3-VL-30B-A3B) and each of 5 seeds, run:
```bash
cd vote_prediction
python qstn_vote_prediction.py --model_id <huggingface_model_id> --seed <42|43|44|45|46>
```
This covers both task phrasings, all 4 persona formats, and both JSON/free-text output modes in one
pass, writing `results/pre_election/results_<model>_seed<seed>.csv`. `slurm/run_qstn_vote_prediction.sh` loops over
all models/seeds as a Slurm batch job (or runs `<model_id> <seed>...` given as arguments). Use `--debug` for a quick 32-persona smoke test.

### 3. Annotate free-text responses (GPU, optional)
Extract vote choice from the open-ended (free-text) responses with an LLM judge (GLM-4.7-Flash):
```bash
cd vote_prediction
python qstn_response_annotation.py --model_id zai-org/GLM-4.7-Flash --source_model <model>
```
or `slurm/run_qstn_annotation.sh <model>`. This reads the per-seed CSVs of `<model>` (e.g. `Llama-3.1-8B-Instruct`)
from step 2 and merges the judged answers into `results/pre_election/annotated_GLM-4.7-Flash_<model>.csv`.
Without `--source_model`, all models are annotated into a single `annotated_GLM-4.7-Flash.csv`.
The evaluation notebooks read the zstd-compressed files, e.g.
`zstd annotated_GLM-4.7-Flash_<model>.csv -o annotated_GLM-4.7-Flash_<model>.zst`.

### 4. Compute evaluation metrics
Run `eval_survey_responses_subgroup.ipynb`. It loads the annotated responses and the preprocessed ANES
data, computes weighted F1 and total variation distance (TVD) — both overall and per demographic
subgroup, with/without ANES survey weights — for each of the 144 stochastic configurations across 5
seeds, and saves `results/pre_election/evaluated_results_subgroup_nonadjusted.csv` plus the reliability plot
(`results/pre_election/reliability.pdf`, Fig. 4).

### 5. Run the regressions and generate figures
- `regression.ipynb` — descriptive results (Table 1), best/worst configuration, and OLS regressions of
  F1/TVD on the 5 design choices, producing the overall/gender/ideology lollipop plots (Figs. 5 & 8).
- `regression_llm_interactions.ipynb` — the same regressions with LLM × design-choice interactions
  (Figs. 9 & 10).
- `regression_individual.ipynb` — individual-level classification reports and the ANES-vs-simulated
  vote-share comparison by ideology subgroup (Fig. 7).

All figures are saved to `results/pre_election/`.

In `regression.ipynb` / `regression_llm_interactions.ipynb`, set `main_metric` /
`subgroup_metric` to `'weighted_f1'` or `'weighted_tvd'` to switch between the two metric variants of
each figure.

## Notes
- All exact prompts (system prompts, persona templates, task phrasings) are defined in
  `vote_prediction/templates.py` and `vote_prediction/qstn_vote_prediction.py`, matching Appendix 9.2.3
  of the paper.
- The full simulation (step 2) is GPU- and time-intensive (2.4M generations across 6 models); the
  Slurm scripts assume a cluster with vLLM/CUDA set up. Most users should start at step 4 using the
  precomputed `results/pre_election/annotated_GLM-4.7-Flash_<model>.zst`.
- The information on which variables are used can be found in the Appendix 9.2.2 (Table 4).