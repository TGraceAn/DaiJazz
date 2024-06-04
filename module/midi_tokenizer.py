import json
import regex as re
from functools import lru_cache

TOKENS = [r'\[WHAT\]\:\d+\_\d+', r'\[HOW\]\:\d+', r'\[WHEN\]\:', r'\[INSTRUMENT\]\:\d+',
                r'\[CC_V\]\:\d+', r'\[PT_V\]\:\d+', r'\[AT_V\]\:\d+', r'\[P_V\]\:', r'\[INSTRUMENT\]\:\d+\.',
                r'\[PIECE\_START\]', r'\[TRACK\_START\]', r'\[TRACK\_END\]', r'\[FILL\_PLACEHOLDER\]',
                r'\[FILL\_START\]', r'\[FILL\_END\]']


class Tokenizer:
    def __init__(self, encoder, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors 
        self.cache = {}

        #paterm to catch the special tokens
        self.pattern = re.compile('|'.join(token for token in TOKENS))

    @lru_cache()
    def encode(self, text):
        tokens = []
        start = 0
        for match in self.pattern.finditer(text):
            # Encode the text before the matched token
            tokens.extend([self.encoder[char] for char in text[start:match.start()]])
            # Encode the matched token
            tokens.append(self.encoder[match.group()])
            start = match.end()
        # Encode the remaining text after the last matched token
        tokens.extend([self.encoder[char] for char in text[start:]])
        return tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        return text
    
    def peek_encoder(self):
        print(self.encoder)

def get_tokenizer():
    with open('encoder.json', 'r') as f:
        encoder = json.load(f)

    return Tokenizer(
        encoder=encoder
    )

if __name__ == '__main__':
    demo_text = """[PIECE_START]
[TRACK_START]
[INSTRUMENT]:0
[WHAT]:2_7 [CC_V]:90 [WHEN]:0
[WHAT]:2_10 [CC_V]:64 [WHEN]:0
[WHAT]:0_64 [HOW]:92 [WHEN]:34816
[WHAT]:0_64 [HOW]:0 [WHEN]:672"""
    tokenizer = get_tokenizer()
#     tokenizer.peek_encoder()

    encoded = tokenizer.encode(demo_text)
    print(tokenizer.decode(encoded) == demo_text)