# TS2E ANES Simulation
Implement design choices in US vote prediction using the [QSTN framework](https://github.com/dess-mannheim/QSTN/)

## Getting Started

First, install the dependencies (includes vllm, torch, etc.)
```bash
pip install qstn[vllm]
```

Then, download and preprocess the ANES data, as shown in `preprocess_ANES.ipynb`.

Next, run the survey inference with `vote_prediction/qstn_vote_prediction.py`.

`vote_prediction/qstn_response_annotation.py` can be used for annotating open-ended responses with LLM-as-a-judge.

Finally, `eval_survey_responses.ipynb` automates evaluation of all simulation specifications and provides plots.
Already evaluated survey predictions are also provided in `results/evaluated_results.csv`.
