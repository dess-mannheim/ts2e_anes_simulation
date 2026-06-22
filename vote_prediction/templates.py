from typing import Tuple, Dict

import pandas as pd


### Persona templates
def create_personas(anes_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:

    # NOTE: personas with item-level missingness can be filtered out later, they don't have to be run separately

    first_person_personas = {} # follows the Argyle et al. (2023) template
    interview_personas = {} # follows the interview format from Lutz et al. (2025)

    # NOTE: The interview questions currently do not follow the ANES wording.
    # This is partially due to the fact that different wordings where used in different survey modes, showcards, etc.

    for id, row in anes_df.iterrows():
        first_person_persona = ""
        interview_persona = ""
        if "race" in row.index and pd.notna(row.race):
            first_person_persona += f"Racially, I am {row.race}. " # NOTE: always keep the space at the end for concatenation
            interview_persona += "Interviewer: What is your race?\n"
            interview_persona += f"Interviewee: I am {row.race}.\n"
        if "discuss_politics" in row.index and pd.notna(row.discuss_politics):
            interview_persona += "Interviewer: Do you like to discuss politics with your family and friends?\n"
            if row.discuss_politics:
                first_person_persona += "I like to discuss politics with my family and friends. "
                interview_persona += "Interviewee: Yes, I like to discuss politics with my family and friends.\n"
            else:
                first_person_persona += "I never discuss politics with my family or friends. "
                interview_persona += "Interviewee: No, I never discuss politics with my family and friends.\n"
        if "ideology" in row.index and pd.notna(row.ideology):
            first_person_persona += f"Ideologically, I am {row.ideology}. "
            interview_persona += "Interviewer: What is your political ideology, from extremely liberal to extremely conservative?\n"
            interview_persona += f"Interviewee: Ideologically, I am {row.ideology}.\n"
        if "party" in row.index and pd.notna(row.party):
            first_person_persona += f"Politically, I am an {row.party}. "
            interview_persona += "Interviewer: Which party do you identify with politically?\n"
            interview_persona += f"Interviewee: Politically, I am an {row.party}.\n"
        if "church_goer" in row.index and pd.notna(row.church_goer):
            first_person_persona += f"I {row.church_goer}. "
            interview_persona += "Interviewer: Do you ever attend church?\n" # NOTE: the ANES also allows for other religious services
            interview_persona += f"Interviewee: I {row.church_goer}.\n"
        if "age" in row.index and pd.notna(row.age):
            first_person_persona += f"I am {int(row.age)} years old. "
            interview_persona += "Interviewer: How old are you?\n"
            interview_persona += f"Interviewee: I am {int(row.age)} years old.\n"
        if "gender" in row.index and pd.notna(row.gender):
            first_person_persona += f"I am {row.gender}. " # NOTE: Argyle et al. (2023) did not include nonbinary
            interview_persona += "Interviewer: What is your gender?\n"
            interview_persona += f"Interviewee: I am {row.gender}.\n"
        if "political_interest" in row.index and pd.notna(row.political_interest):
            first_person_persona += f"I am {row.political_interest} interested in politics. "
            interview_persona += "Interviewer: Are you interested in politics?\n"
            interview_persona += f"Interviewee: I am {row.political_interest} interested in politics.\n"
        if "patriotism" in row.index and pd.notna(row.patriotism): # NOTE: this question was dropped from the 2020 and 2024 versions of the ANES
            first_person_persona += f"It makes me feel {row.patriotism} to see the American flag. "
            interview_persona += "Interviewer: How does it make you feel to see the American flag?\n"
            interview_persona += f"Interviewee: It makes me feel {row.patriotism} to see the American flag.\n"
        if "state" in row.index and pd.notna(row.state):
            first_person_persona += f"I am from {row.state}. "
            interview_persona += "Interviewer: Which state are you from?\n"
            interview_persona += f"Interviewee: I am from {row.state}.\n"

        first_person_personas[id] = first_person_persona
        interview_personas[id] = interview_persona
    
    return first_person_personas, interview_personas
