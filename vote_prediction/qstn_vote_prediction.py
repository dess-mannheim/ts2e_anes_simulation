import argparse
import random

import pandas as pd
from qstn.inference.local_inference import default_model_init
from qstn.inference.response_generation import JSONSingleResponseGenerationMethod, LogprobResponseGenerationMethod
from qstn.prompt_builder import LLMPrompt
from qstn.utilities import placeholder, create_one_dataframe
from qstn.utilities.survey_objects import AnswerOptions, AnswerTexts
from qstn.survey_manager import conduct_survey_single_item
from qstn.parser import raw_responses, parse_logprobs, parse_json

from templates import create_personas


### Experiment settings
DEBUG_SAMPLE_SIZE = 32
SAVE_PATH = "../results/{debug}results_{model_str}_seed{seed}.csv"

def main():

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--model_id', type=str, help='LLM model name from huggingface')
  arg_parser.add_argument('--debug', type=bool, default=False,
                      const=True, nargs='?', # allow for just setting "--debug" to evaluate to True
                      help='Run in debug mode')
  arg_parser.add_argument('--n_samples', type=int, default=1,
                      help='How many responses to sample for each setting')
  arg_parser.add_argument('--seed', type=int, default=42, help='random seed')

  args = arg_parser.parse_args()

  random.seed(args.seed)

  model_kwargs = {"max_model_len": 15000, "seed": args.seed}
  chat_template_kwargs = {'enable_thinking': False}
  generation_kwargs = {'max_tokens': 7000, 'seed': args.seed}

  if "gpt-oss" in args.model_id:
    print("Disabeling --async-scheduling for vllm due to a issue with GPT-OSS models: https://github.com/vllm-project/vllm/issues/22513")
    model_kwargs = model_kwargs | {"async_scheduling": False}


  ### Set up the survey
  vote_questionnaire = pd.DataFrame([
    {"questionnaire_item_id": 1, "question_content": "Will you vote in the 2024 U.S. presidential election and if so for whom?"},
    {"questionnaire_item_id": 2, "question_content": "Please tell me if you will vote and if so for whom."},
  ])

  _options = AnswerTexts(['Harris', 'Trump', 'Non-voter'])
  response_option_formats = {
    'openResponse': AnswerOptions(_options), # open-ended responses do not have a response generation method
    'jsonResponse': AnswerOptions(_options, response_generation_method = JSONSingleResponseGenerationMethod()),
    #'logprobResponse': AnswerOptions(_options, response_generation_method = LogprobResponseGenerationMethod())
  }

  anes_df = pd.read_csv('../data/2024_anes_preprocessed.csv') # NOTE: contains item-level missingness
  if args.debug: anes_df = anes_df.sample(DEBUG_SAMPLE_SIZE, random_state=args.seed)

  # Full personas based on available items — complete cases can be filtered later
  first_person_personas, interview_personas = create_personas(anes_df)

  # Short personas based on only the subset of demographic variables
  demographic_df = anes_df[["age", "race", "gender", "state"]]
  demogr_first_person_personas, demogr_interview_personas = create_personas(demographic_df)

  persona_formats = {
    'firstPerson': first_person_personas,
    'interview': interview_personas,
    'demogrFirstPerson': demogr_first_person_personas,
    'demogrInterview': demogr_interview_personas,
  }

  SIM_SYSTEM_PROMPT = f"You are a political scientist predicting vote choice in the 2024 U.S. presidential election. {placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}"

  interviews: list[LLMPrompt] = []

  for response_type, response_options in response_option_formats.items():
    for persona_format, personas in persona_formats.items():
      for persona_id, persona in personas.items():
        for sample in range(args.n_samples):

          user_prompt = persona

          # For interview-style prompts, follow the same format as in the persona template
          if 'interview' in persona_format.lower():
            user_prompt += f"Interviewer: {placeholder.PROMPT_QUESTIONS}\nInterviewee:"
          # For self-description prompts, contextualize the question
          elif 'firstperson' in persona_format.lower():
            user_prompt += f"When I'm being asked '{placeholder.PROMPT_QUESTIONS}', I respond with"
          else:
            user_prompt += f"{placeholder.PROMPT_QUESTIONS}"

          interview = LLMPrompt(
            questionnaire_name = f'{response_type}_{persona_format}_{persona_id}_{sample}',
            questionnaire_source = vote_questionnaire,
            system_prompt = SIM_SYSTEM_PROMPT,
            prompt = user_prompt,
          )

          interview.prepare_prompt(
            question_stem=f"{placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}",
            answer_options = response_options,
            randomized_item_order = False,
          )

          interviews.append(interview)

  ### Start up the vllm model
  model = default_model_init(
      model_id = args.model_id,
      gpu_memory_utilization = 0.95,
      #tensor_parallel_size = 2, # NOTE: number of GPUs is automatically identified
      **model_kwargs
  )

  ### Run the survey simulation
  results = conduct_survey_single_item(
    model,
    llm_prompts = interviews,
    print_conversation = args.debug,
    chat_template_kwargs=chat_template_kwargs,
    **generation_kwargs
  )

  ### Parse the results
  parsed_results = {}

  for result in results:
    # Use logprob parser for logprob results
    if "logprobResponse" in result.questionnaire.questionnaire_name:
      _parsed = parse_logprobs(
        survey_results = [result],
        allowed_choices = {
          'Harris': ['Harris', 'Kamala Harris', 'harris', 'kamala harris'],
          'Trump': ['Trump', 'Donald Trump', 'trump', 'donald trump'],
          'Non-voter': ['Non-voter', 'non-voter'],
        }
      )
      parsed_results = parsed_results | _parsed
    # TODO: implement parsing with LLM as a judge
    # Use LLM as a judge for open-ended responses
    #elif "openResponse" in result.questionnaire.questionnaire_name:
    # ...
    elif "jsonResponse" in result.questionnaire.questionnaire_name:
      parsed_results = parsed_results | parse_json([result])
    else:
      parsed_results = parsed_results | raw_responses([result])

  parsed_results_df = create_one_dataframe(parsed_results)

  parsed_results_df.to_csv(
    SAVE_PATH.format(
      model_str = args.model_id.split('/')[1],
      debug = 'debug_' if args.debug else '',
      seed = args.seed
    ),
    index=False,
    encoding="utf-8",
    errors="replace"
  )

if __name__ == "__main__":
    main()
