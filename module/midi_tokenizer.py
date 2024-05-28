import json
import regex as re
from functools import lru_cache

#Tokens
TOKENS = ['[WHAT]', '[WHICH]', '[HOW]', '[WHEN]', '[PIECE_START]','[TRACK_START]', '[TRACK_END]',
                '[INSTRUMENT]','[FILL_PLACEHOLDER]','[FILL_START]', '[FILL_END]']

class Tokenizer:
    def __init__(self, encoder, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors 
        self.cache = {}

        #paterm to catch the special tokens
        self.pattern = re.compile('|'.join(re.escape(token) for token in TOKENS))

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
[WHAT]:2 [WHICH]:121 [HOW]:0 [WHEN]:0
[WHAT]:2 [WHICH]:67 [HOW]:0 [WHEN]:4
[WHAT]:2 [WHICH]:64 [HOW]:0 [WHEN]:2
[WHAT]:2 [WHICH]:100 [HOW]:0 [WHEN]:2
[WHAT]:2 [WHICH]:101 [HOW]:0 [WHEN]:2
[WHAT]:2 [WHICH]:6 [HOW]:12 [WHEN]:2
[WHAT]:2 [WHICH]:11 [HOW]:127 [WHEN]:2
[WHAT]:2 [WHICH]:1 [HOW]:0 [WHEN]:2
[WHAT]:2 [WHICH]:11 [HOW]:127 [WHEN]:4
[WHAT]:4 [HOW]:0 [WHEN]:4
[WHAT]:2 [WHICH]:0 [HOW]:0 [WHEN]:1128
[INSTRUMENT] 0."""
    tokenizer = get_tokenizer()
    encoded = tokenizer.encode(demo_text)

    print(tokenizer.decode(encoded))