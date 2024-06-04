import os
import json
import re

REGEX_TOKENS = [r'\[WHAT\]\:\d+\_\d+', r'\[HOW\]\:\d+', r'\[WHEN\]:', r'\[INSTRUMENT\]\:\d+',
                r'\[CC_V\]\:\d+', r'\[PT_V\]\:\d+', r'\[AT_V\]\:\d+', r'\[P_V\]:', r'\[INSTRUMENT\]\:\d+\.',
                r'\[PIECE\_START\]', r'\[TRACK\_START\]', r'\[TRACK\_END\]', r'\[FILL\_PLACEHOLDER\]',
                r'\[FILL\_START\]', r'\[FILL\_END\]']

NUMBER_PATTERN = r'\d+'
pattern = re.compile('|'.join(REGEX_TOKENS))
number_pattern = re.compile(NUMBER_PATTERN)

TOKEN_NAMES = '[WHAT]:[HOW]:[WHEN]:[INSTRUMENT]:.[CC_V]:[PT_V]:[AT_V]:[P_V]:[PIECE_START][TRACK_START][TRACK_END][FILL_PLACEHOLDER][FILL_START][FILL_END]'
remove_set = set(TOKEN_NAMES)

def get_tokens(path):
    # Create a tokenizer file
    encoded = {}

    if not os.path.exists('encoder.json'):
        os.system('touch encoder.json')

    # Read the txt files
    all_tokens = set()
    for file in os.listdir(path):
        if file.endswith('.txt'):
            file_name = file[:-4]
            with open(f'{path}/{file_name}.txt', 'r') as f:
                text = f.read()
                tokens = pattern.findall(text)
                all_tokens.update(tokens)

                # Find numbers after [WHEN]:
                when_tokens = re.findall(r'\[WHEN\]:(\d+)', text)
                all_tokens.add('[WHEN]:')
                for num in when_tokens:
                    all_tokens.update(list(num))
                    
                # Find numbers after [P_V]:
                pv_tokens = re.findall(r'\[P\_V\]:(\d+)', text)
                all_tokens.add('[P_V]:')
                for num in pv_tokens:
                    all_tokens.update(list(num))

                # Find single characters not part of regex tokens
                single_chars = set(re.sub(r'|'.join(REGEX_TOKENS), '', text))
                all_tokens.update(single_chars)

    # Remove duplicate single characters from the token set
    unique_tokens = all_tokens - remove_set

    # Encode tokens
    for i, token in enumerate(sorted(set(unique_tokens))):
        encoded[token] = i

    with open('encoder.json', 'w') as f:
        json.dump(encoded, f)

if __name__ == '__main__':
    get_tokens(path='/Users/thienannguyen/Desktop/7th_Test/txt_aug')