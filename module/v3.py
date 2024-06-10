import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import regex as re
from midi_tokenizer import get_tokenizer
from tqdm import tqdm
import math


# relative context
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions? --> (Probably need longer)

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda'
eval_iters = 200

n_embd = 2130 

n_head = 10 #10 --> each head is 2130/10 = 213
n_layer = 6 #12
dropout = 0.1
# ------------


torch.manual_seed(1162003)
tokenizer = get_tokenizer()

# Load MIDI data, each track is a piece of music in this context
tracks = []
val_tracks = []

error_tracks = []

#regex to remove numbers after [WHEN]:
pattern = r'\[WHEN\]:(\d+)'

# an array to store all the numbers after [WHEN]:
when_tokens = []

# for file in os.listdir('txt_out'):
#     if file.endswith('.txt'):
#         with open(f'txt_out/{file}', 'r') as f:
#             text = f.read()
#             # Remove numbers after [WHEN]:
#             when_tokens = re.findall(r'\[WHEN\]:(\d+)', text)
#             text = re.sub(pattern, '[WHEN]:', text)

#             # Remove spaces
#             text = re.sub(r' ', '', text)
#             try:
#                 tokens = tokenizer.encode(text)
#                 tracks.append(tokens)
#             except:
#                 error_tracks.append(file)
    
with open(f'txt_out/Osaka, November 8, 1976 (Part 1).txt') as f:
    text = f.read()
    # Remove numbers after [WHEN]:
    when_tokens = re.findall(r'\[WHEN\]:(\d+)', text)
    text = re.sub(pattern, '[WHEN]:', text)

    # Remove spaces
    text = re.sub(r' ', '', text)
    tracks.append(tokenizer.encode(text))
    

vocab_size = len(tokenizer.encoder)


data = torch.tensor(tracks[0], dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = scale
        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=False)
        self.fc = nn.Linear(d_head * n_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.zeros((n_head, d_head)))
            self.r_w_bias = nn.Parameter(torch.zeros((n_head, d_head)))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def forward(self, x, r_r_buckets, r_w_buckets, r_emb, attn_mask=None):
        qkv = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(*q.size()[:2], self.n_head, self.d_head).transpose(1, 2)
        k = k.view(*k.size()[:2], self.n_head, self.d_head).transpose(1, 2)
        v = v.view(*v.size()[:2], self.n_head, self.d_head).transpose(1, 2)

        r_r_emb = r_emb(r_r_buckets)
        r_w_emb = r_emb(r_w_buckets)

        qr = torch.einsum('bxhd,hd->bxh', q, self.r_r_bias)
        qr = torch.einsum('bxh,bhd->bxhd', qr, r_r_emb)
        kr = torch.einsum('byhd,hd->byh', k, self.r_w_bias)
        kr = torch.einsum('byh,bhd->byhd', kr, r_w_emb)

        if self.scale:
            q = q / math.sqrt(self.d_head)

        attn_weights = torch.einsum('bxhd,byhd->bxyh', q, k)
        attn_weights = attn_weights + qr[:, :, None] + kr[:, None, :, :]

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=3)
        attn_weights = self.dropout(attn_weights)
        out = torch.einsum('bxyh,byhd->bxhd', attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(*out.size()[:2], self.d_model)
        out = self.fc(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Linear(n_embd, 4 * n_embd)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(4 * n_embd, n_embd)),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(n_head, d_model, d_head, dropout, r_w_bias, r_r_bias, scale)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, r_r_buckets, r_w_buckets, r_emb, attn_mask=None):
        x = x + self.attn(self.ln1(x), r_r_buckets, r_w_buckets, r_emb, attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class DaiJazz_Piano(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.r_w_bias = nn.Parameter(torch.zeros((n_head, n_embd // n_head)))
        self.r_r_bias = nn.Parameter(torch.zeros((n_head, n_embd // n_head)))
        self.r_emb = nn.Embedding(2 * block_size - 1, n_embd // n_head)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, n_embd // n_head, dropout, self.r_w_bias, self.r_r_bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        r_r_buckets = self.relative_buckets(T, rel_type='relative')
        r_w_buckets = self.relative_buckets(T, rel_type='window')
        for block in self.blocks:
            x = block(x, r_r_buckets, r_w_buckets, self.r_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def relative_buckets(self, T, rel_type='relative'):
        if rel_type == 'relative':
            buck = torch.arange(-T + 1, T, device=device)
        else:
            buck = torch.arange(T, device=device)
        buck = buck.unsqueeze(0).expand(self.n_head, -1)
        return buck

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = DaiJazz_Piano()

def train(model):
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    previous = None
    for iter in tqdm(range(max_iters)):
        losses = estimate_loss()


        torch.save(model, 'last.pth')
        # every once in a while evaluate the loss on train and val sets

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if previous == None or previous >= float(losses['val']):
                previous = float(losses['val'])
                print('Saving best model')
                torch.save(model, 'best.pth')
            


        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return m


# generate from the model
m = train(model)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))