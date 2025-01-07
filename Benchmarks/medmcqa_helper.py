import random
from textwrap import dedent  # Remove leading whitespace from multiline strings
from jellyfish import levenshtein_distance

def get_answer(entry):
    # entry['cop'] is an integer in the range 0..3 that
    # denotes the correct option (a, b, c or d).
    options = {0:'opa', 1:'opb', 2:'opc', 3:'opd'}
    correct_option = options[entry['cop']]
    answer = entry[correct_option]
    return answer

def check_answer(entry, answer):
    possible_answers = [
        entry['opa'],
        entry['opb'],
        entry['opc'],
        entry['opd'],
    ]
    lev_distances = [levenshtein_distance(answer, s)
                     for s in possible_answers]
    closest_answer_index = lev_distances.index(min(lev_distances))
    is_correct = (closest_answer_index == entry['cop'])
    return is_correct

def add_prompt(entry, tokenizer, include_answer, shuffle_options=False):
    options = [
        entry["opa"],
        entry["opb"],
        entry["opc"],
        entry["opd"]
    ]
    if shuffle_options:
        random.shuffle(options)
    messages = [
        {'role': 'user', 'content': dedent(f'''\
            You are a medical student taking a multiple-choice exam. Four options are provided for each question. Only one of these options is the correct answer.
            Question: {entry["question"]}
            Options:
            1. {options[0]}
            2. {options[1]}
            3. {options[2]}
            4. {options[3]}
            Solve this multiple-choice exam question and provide the correct answer.''')
        }
    ]
    if include_answer:
        answer = get_answer(entry)
        messages.append(
            {'role': 'assistant', 'content': f'Answer: {answer}'}
        )
    entry['text'] = tokenizer.apply_chat_template(messages, tokenize=False)
    return entry
