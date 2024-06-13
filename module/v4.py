import torch
import torch.nn as nn
from torch.nn import functional as F
from midi_tokenizer import get_tokenizer
from dataclasses import dataclass
import os
import regex as re
import time

tokenizer = get_tokenizer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = len(tokenizer.encoder)
    n_layer: int = 6 #6 hidden layers
    n_head: int = 16
    n_embd: int = 256

#MLP for the transformer
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.act = F.gelu

    def forward(self, x):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return x

# RelativeMultiHeadAttention from Music Transformer
class RelativeMultiHeadAttention(nn.Module):
    pass

# Can you RelativeMultiHeadAttention or MultiHeadSelfAttention?
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Multihead attention layer should be self-attention
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=0.2) #Dropout is 0.2
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        #residual connection for communication
        x = x + self.attn(self.ln_1(x))
        #residual connection for making computation
        x = x + self.mlp(self.ln_2(x))
        return x

#Basically GPT-2 architecture
class DaiJazz_Piano(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # weight embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # transformer hidden blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Init weights
        self.apply(self._init_weights)
        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    # This is not my code so I'm not sure if this is correct
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # residual connection control
            std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        if targets is not None:
            
            #try loss in NLL
            loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


#-------------------------
#Dataloader for the model
class DataLoader:
    def __init__(self, B, T):
        #B: batch size, T: sequence length
        self.B = B
        self.T = T
        self.tracks = []
        
        for file in os.listdir('txt_out'):
            if file.endswith('.txt'):
                with open(f'txt_out/{file}') as f:
                    text = f.read()
                    #remove spaces and newlines
                    text = text.replace(' ', '').replace('\n', '')
                    #remove [WHEN]: tokens
                    text = re.sub(r'\[WHEN\]\:', '', text)
                    #tokenize
                    tokens = tokenizer.encode(text)
                    self.tracks.append(tokens)
                
        #split into train and validation (90% train, 10% validation)
        self.tracks = self.tracks[:int(0.9*len(self.tracks))]
        self.val = self.tracks[int(0.9*len(self.tracks)):]
        
        #concatenate all tracks
        self.tracks = torch.cat(self.tracks)
        self.val = torch.cat(self.val)

        #return tensors
        self.tokens = torch.tensor(self.tracks, dtype=torch.long)
        self.val_tokens = torch.tensor(self.val, dtype=torch.long)

        print(f'Loaded {len(self.tracks)} tracks, {len(self.tokens)} tokens')
        print(f'1 Epoch = {len(self.tokens) // (B * T)} batches')

        self.val_steps = len(self.val_tokens) // (B * T)

        #initialize the pointers
        self.i = 0
        self.train_epoch = 0


    def __iter__(self):
        B, T = self.B, self.T
        buf = self.tokens[self.i:self.i + B*T]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.i += B*T
        if self.i + (B * T + 1) >= len(self.tokens):
            self.i = 0
            self.train_epoch += 1
        return x, y
    
    def val_loader(self, step):
        B, T = self.B, self.T
        pointer = step * B * T
        buf = self.val_tokens[pointer:pointer + B*T]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        return x, y
    
    # inplace of __iter__ --> Use when need to do gradient accumulation
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.i:self.i + B*T]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.i += B*T
        if self.i + (B * T + 1) >= len(self.tokens):
            self.i = 0
            self.train_epoch += 1
        return x, y

#--------------------------------
def get_val_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range (data_loader.val_steps):
            x, y = data_loader.val_loader(i)
            logits, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(data_loader.val_tokens)

#Cosine learning rate decay
def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1)/warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio) * 3.14159))
    return min_lr + coeff * (max_lr - min_lr)

def training_pipeline():
    config = ModelConfig()
    model = DaiJazz_Piano(config)
    model.to(device)

    B = 16
    T = 512

    # # This is for gradient accumulation
    # total_batch_size = 524288
    # assert total_batch_size % (B * T) == 0, 'Batch size must divide total batch size'
    # gradient_accumulation_steps = total_batch_size // (B * T)

    data_loader = DataLoader(B=B, T=T)

    #cosine learning rate decay
    max_lr = 6e-4
    min_lr = max_lr*0.1
    # num_tokens = data_loader.tokens

    max_steps = len(data_loader.tokens) // data_loader.B * data_loader.T
    warmup_steps = max_steps // 10

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    start_time = time.time()

    while data_loader.train_epoch < 20:
        for i, (x, y) in enumerate(data_loader):
            current_epoch = data_loader.train_epoch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #Warmup learning rate
            if data_loader.train_epoch == 0:
                lr = get_lr(i, max_lr, min_lr, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            optimizer.step()
            # synchronize
            torch.cuda.synchronize()

            if i % 10 == 0:
                end_time = time.time()
                print(f'Epoch {data_loader.train_epoch}, Iteration {i}, Loss {loss.item()}, Average time {(end_time - start_time)*1000/10}ms, Norm {norm:.4f}, LR {lr:.4f}')
                start_time = time.time()

            previous_val_loss = None
            # Evaluate the model at the end/beginning of each epoch (I'm not sure)
            if current_epoch != data_loader.train_epoch:
                val_loss = get_val_loss(model, data_loader.val_loader())
                print(f'Validation loss: {val_loss}')
                if previous_val_loss == None or previous_val_loss >= val_loss:
                    previous_val_loss = val_loss
                    print('Saving best model...')
                    torch.save(model.state_dict(), 'best.pth')

                print('Saving last model...')
                torch.save(model.state_dict(), 'last.pth')
                model.train()
                current_epoch = data_loader.train_epoch

    #Save the last ever model
    torch.save(model.state_dict(), 'last.pth')
    print('Training done!')

if __name__ == '__main__':
    training_pipeline()



                    

    
