import torch
import torch.nn as nn
from torch.nn import functional as F
from midi_tokenizer import get_tokenizer
import os

# relative context
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions? --> (Probably need longer)

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 384 #410
n_head = 6 #10
n_layer = 6 #12
dropout = 0.2
# ------------


torch.manual_seed(1162003)

# Load all the text file
texts = []
for file in os.listdir('txt_aug'):
    if file.endswith('.txt'):
        with open(f'txt_aug/{file}', 'r') as f:
            text = f.read()
            texts.append(text)

tokenizer = get_tokenizer()
vocab_size = len(tokenizer.encoder)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
data = data[2:] # remove the first two tokens [PIECE_START] and \n

#Split into tracks
tracks = []
current_track = []
for token in data.tolist():
    if token == 20:  # Token "20" represents [TRACK_START]
        if current_track:
            tracks.append(torch.tensor(current_track[1:]))
            current_track = []
    else:
        current_track.append(token)

# Append the last track
if current_track:
    current_track = current_track[1:]
    tracks.append(torch.tensor(current_track))

track_heads = len(tracks)

# time_embed = 


