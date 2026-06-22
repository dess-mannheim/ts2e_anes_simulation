import argparse
import os
import random

import pandas as pd
from qstn.inference.local_inference import default_model_init
from qstn.inference.response_generation import JSONSingleResponseGenerationMethod
from qstn.prompt_builder import LLMPrompt
from qstn.utilities import placeholder, create_one_dataframe
from qstn.utilities.survey_objects import AnswerOptions, AnswerTexts
from qstn.survey_manager import conduct_survey_single_item
from qstn.parser import parse_json


### Experiment settings
DEBUG_SAMPLE_SIZE = 8
SEED = 42 # random seed
SAVE_PATH = "../results/{debug}annotated_{model_str}.csv"
LOAD_PATH = "../results/results_{model_str}{seed_suffix}.csv"
SOURCE_SEEDS = [42, 43, 44, 45, 46]

LOAD_MODELS = [
  # Qwen
  "Qwen3-VL-8B-Instruct",
  "Qwen3-VL-30B-A3B-Instruct",
  # Olmo
  "Olmo-3-7B-Instruct",
  "Olmo-3.1-32B-Instruct",
  # Meta
  "Llama-3.1-8B-Instruct",
  "Llama-3.3-70B-Instruct",
]

### Prompts for LLM-as-a-judge
PARSER_SYSTEM_PROMPT = f"""You are an expert annotator labeling statements about vote choice in the 2024 U.S. presidential election. {placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"""

PARSER_USER_PROMPT = """Below is a STATEMENT that responds to the question: "{question_content}"
Does the following STATEMENT express preference for exactly one of the following answer options? {prompt_options} If yes, which option is preferred?

STATEMENT:

{llm_response}"""


random.seed(SEED)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model_id', type=str, help='LLM model name from huggingface')
arg_parser.add_argument('--debug', type=bool, default=False,
                    const=True, nargs='?', # allow for just setting "--debug" to evaluate to True
                    help='Run in debug mode')

args = arg_parser.parse_args()

model_kwargs = {"max_model_len": 15000}
chat_template_kwargs = {'enable_thinking': False}
generation_kwargs = {'max_tokens': 7000, 'seed': SEED}

if "gpt-oss" in args.model_id:
  print("Disabeling --async-scheduling for vllm due to a issue with GPT-OSS models: https://github.com/vllm-project/vllm/issues/22513")
  model_kwargs = model_kwargs | {"async_scheduling": False}


### Set up the survey
vote_questionnaire = pd.DataFrame([
  {"questionnaire_item_id": 1, "question_content": "Will you vote in the 2024 U.S. presidential election and if so for whom?"},
  #{"questionnaire_item_id": 2, "question_content": "Please tell me if you will vote and if so for whom."}, # only annotate with one question
])

_options = AnswerTexts(['Harris', 'Trump', 'Non-voter'])
response_options = AnswerOptions(_options, response_generation_method = JSONSingleResponseGenerationMethod())


save_path = SAVE_PATH.format(
  model_str=args.model_id.split('/')[1],
  debug='debug_' if args.debug else ''
)
existing_annotated_df = pd.read_csv(save_path, low_memory=False) if os.path.exists(save_path) else None

if existing_annotated_df is not None:
  already_annotated_keys = set(
    existing_annotated_df[
      existing_annotated_df.response_type == 'openResponse'
    ][['model_id', 'seed', 'questionnaire_name', 'questionnaire_item_id']]
    .astype(str).agg('_'.join, axis=1)
  )
else:
  already_annotated_keys = set()

interviews: list[LLMPrompt] = []
original_response_dfs = []

for original_model in LOAD_MODELS:
  for seed in SOURCE_SEEDS:
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    path = LOAD_PATH.format(model_str=original_model, seed_suffix=seed_suffix)
    if not os.path.exists(path):
      continue  # tolerate missing seed files

    response_df = pd.read_csv(path)

    if args.debug: response_df = response_df.sample(DEBUG_SAMPLE_SIZE, random_state=SEED)

    response_df[["response_type", "persona_format", "persona_id", "sample"]] = response_df.questionnaire_name.str.split('_').to_list()
    response_df['model_id'] = original_model
    response_df['seed'] = seed

    # include seed so identical questionnaire_names from different seeds don't collide
    response_df['combined_questionnaire_name'] = response_df[
      ['model_id', 'seed', 'questionnaire_name', 'questionnaire_item_id']
    ].astype(str).agg('_'.join, axis=1)

    open_response_df = response_df[response_df.response_type == 'openResponse']
    original_response_dfs.append(response_df)

    for idx, response_row in open_response_df.iterrows():

      if response_row['combined_questionnaire_name'] in already_annotated_keys:
        continue

      question_stem = PARSER_USER_PROMPT.format(
        llm_response = response_row['llm_response'],
        question_content = placeholder.QUESTION_CONTENT,
        prompt_options = placeholder.PROMPT_OPTIONS
      )

      interview = LLMPrompt(
        questionnaire_name = response_row['combined_questionnaire_name'],
        questionnaire_source = vote_questionnaire,
        system_prompt = PARSER_SYSTEM_PROMPT,
        prompt = f"{placeholder.PROMPT_QUESTIONS}",
      )

      interview.prepare_prompt(
        question_stem = question_stem,
        answer_options = response_options,
        randomized_item_order = False,
      )

      interviews.append(interview)

### Start up the vllm model
model = default_model_init(
    model_id = args.model_id,
    gpu_memory_utilization = 0.85,
    **model_kwargs
)

### Run the annotation
results = conduct_survey_single_item(
  model,
  llm_prompts = interviews,
  print_conversation = args.debug,
  chat_template_kwargs=chat_template_kwargs,
  **generation_kwargs
)

### Parse the results
parsed_results = parse_json(results)
parsed_results_df = create_one_dataframe(parsed_results)

### Combine parsed results back into original df
original_results = pd.concat(original_response_dfs, ignore_index=True)
original_open_results = original_results[original_results.response_type == 'openResponse'].drop(columns='answer')

parsed_filtered_df = parsed_results_df[['questionnaire_name', 'answer']] # only keep key and parsed answer
parsed_filtered_df = parsed_filtered_df.rename(columns={'questionnaire_name': 'combined_questionnaire_name'}) # already combined

# include previously-annotated open responses so the merge populates their `answer` too
if existing_annotated_df is not None:
  prior_open = existing_annotated_df[existing_annotated_df.response_type == 'openResponse'].copy()
  prior_open['combined_questionnaire_name'] = prior_open[
    ['model_id', 'seed', 'questionnaire_name', 'questionnaire_item_id']
  ].astype(str).agg('_'.join, axis=1)
  parsed_filtered_df = pd.concat(
    [parsed_filtered_df, prior_open[['combined_questionnaire_name', 'answer']]],
    ignore_index=True
  )

combined_results = pd.merge(
  left = original_open_results, # only merge parsed results for open-ended responses
  right = parsed_filtered_df,
  on = 'combined_questionnaire_name',
  how = 'left'
)
combined_results = pd.concat([
  combined_results,
  original_results[original_results.response_type != 'openResponse'] # concat closed responses back together
])

combined_results = combined_results.drop(columns='combined_questionnaire_name')

combined_results.to_csv(
  save_path,
  index=False,
  encoding="utf-8",
  errors="replace"
)
