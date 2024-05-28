import json
import os
import regex as re

TOKENS = ['[WHAT]', '[WHICH]', '[HOW]', '[WHEN]', '[PIECE_START]','[TRACK_START]', '[TRACK_END]',
                '[INSTRUMENT]','[FILL_PLACEHOLDER]','[FILL_START]', '[FILL_END]']

pattern = re.compile('|'.join(re.escape(token) for token in TOKENS))

def get_tokens(path):
    # Create a tokenizer file 
    encoded = {}
    
    if not os.path.exists('encoder.json'):
        os.system('touch encoder.json')

    # Read the txt files
    all_chars = set()
    for file in os.listdir(path):
        if file.endswith('.txt'):
            file_name = file[:-4]
            with open(f'{path}/{file_name}.txt', 'r') as f:
                text = f.read()
                chars = set(text)
                all_chars.update(chars)

    # Remove characters that are part of tokens
    token_chars = set(''.join(TOKENS))
    unique_chars = all_chars - token_chars

    # Encode characters and tokens
    for i, char in enumerate(sorted(unique_chars)):
        encoded[char] = i

    for i, token in enumerate(TOKENS, start=len(unique_chars)):
        encoded[token] = i

    with open('encoder.json', 'w') as f:
        json.dump(encoded, f)

if __name__ == '__main__':
    get_tokens(path='/Users/thienannguyen/Desktop/6th_Test/txt_aug')