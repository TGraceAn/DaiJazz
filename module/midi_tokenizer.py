import json
import regex as re
from functools import lru_cache

TOKENS = [r'\[WHAT\]\:\d+\_\d+', r'\[HOW\]\:\d+', r'\[WHEN\]\:', r'\[INSTRUMENT\]\:\d+',
                r'\[CC_V\]\:\d+', r'\[PT_V\]\:\d+', r'\[AT_V\]\:\d+', r'\[P_V\]\:', r'\[INSTRUMENT\]\:\d+\.',
                r'\[PIECE\_START\]', r'\[TRACK\_START\]', r'\[TRACK\_END\]', r'\[FILL\_PLACEHOLDER\]',
                r'\[FILL\_START\]', r'\[FILL\_END\]', r'\[PIECE\_END\]']


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
[WHAT]:2_64 [CC_V]:127 [WHEN]:7
[WHAT]:2_64 [CC_V]:0 [WHEN]:707
[WHAT]:0_43 [HOW]:54 [WHEN]:18
[WHAT]:0_55 [HOW]:49 [WHEN]:4
[WHAT]:2_64 [CC_V]:127 [WHEN]:78
[WHAT]:0_79 [HOW]:74 [WHEN]:1016
[WHAT]:0_72 [HOW]:82 [WHEN]:0
[WHAT]:0_76 [HOW]:73 [WHEN]:1
[WHAT]:0_69 [HOW]:65 [WHEN]:10
[WHAT]:0_64 [HOW]:67 [WHEN]:2
[WHAT]:0_55 [HOW]:4 [WHEN]:17
[WHAT]:0_43 [HOW]:4 [WHEN]:3
[WHAT]:0_72 [HOW]:4 [WHEN]:233
[WHAT]:0_57 [HOW]:65 [WHEN]:4
[WHAT]:2_64 [CC_V]:0 [WHEN]:4
[WHAT]:0_57 [HOW]:4 [WHEN]:0
[WHAT]:0_69 [HOW]:4 [WHEN]:0
[WHAT]:0_76 [HOW]:4 [WHEN]:0
[WHAT]:0_72 [HOW]:69 [WHEN]:3
[WHAT]:0_76 [HOW]:76 [WHEN]:2"""
    tokenizer = get_tokenizer()
#     tokenizer.peek_encoder()

    encoded = tokenizer.encode(demo_text)
    print(tokenizer.decode(encoded) == demo_text)