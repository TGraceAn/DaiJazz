import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from midi_tokenizer import get_tokenizer
from dataclasses import dataclass
import os
import regex as re
import time
import matplotlib.pyplot as plt
import submitit



tokenizer = get_tokenizer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_save(data, title, xlabel, ylabel, filename, y=None):
    plt.figure()
    y = y if y else range(len(data))

    plt.plot(y, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = len(tokenizer.encoder)
    n_layer: int = 6  # 6 hidden layers
    n_head: int = 64
    n_embd: int = 512


# MLP for the transformer
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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# Can you RelativeMultiHeadAttention or MultiHeadSelfAttention?
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Multihead attention layer should be self-attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # residual connection for communication
        x = x + self.attn(self.ln_1(x))
        # residual connection for making computation
        x = x + self.mlp(self.ln_2(x))
        return x

# Basically GPT-2 architecture
class DaiJazz_Piano(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # weight embeddings
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # position embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # transformer hidden blocks
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer norm
            ln_f=nn.LayerNorm(config.n_embd),
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
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)

        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)

        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None

        if targets is not None:
            # try loss in NLL
            # loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
# --------------------------------

class MIDIDataset(Dataset):
    def __init__(self, root_dir, block_size):
        self.block_size = block_size
        self.data = []
        for file in os.listdir(root_dir):
            if file.endswith('.txt'):
                with open(os.path.join(root_dir, file)) as f:
                    text = f.read().replace(' ', '').replace('\n', '')
                    text = re.sub(r'\[WHEN\]\:', '', text)
                    tokens = tokenizer.encode(text)
                    self.data.extend(tokens)
        self.data = torch.tensor(self.data, dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

root_dir = '../../Test_Linux/txt_out_2'
T = 512
dataset = MIDIDataset(root_dir, T)

batch_size = 64

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# --------------------------------
# Cosine learning rate decay
def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio) * 3.14159))
    return min_lr + coeff * (max_lr - min_lr)


# --------------------------------
# #distributed training
# #run instructions: torchrun --standalone --nproc_per_node=4 module/v6.py

# from torch.distributed import init_process_group, destroy_process_group

# ddp = int(os.environ.get('RANK', -1)) != -1
# if ddp:
#     assert torch.cuda.is_available(), "Distributed training requires CUDA"
#     init_process_group(backend='nccl')
#     ddp_rank = int(os.environ['RANK'])
#     ddp_local_rank = int(os.environ['LOCAL_RANK'])
#     ddp_world_size = int(os.environ['WORLD_SIZE'])
#     device = f'cuda:{ddp_local_rank}'
#     torch.cuda.set_device(device)
#     master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
# else:
#     # vanilla, non-DDP run
#     ddp_rank = 0
#     ddp_local_rank = 0
#     ddp_world_size = 1
#     master_process = True
#     # attempt to autodetect device
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = "mps"
#     print(f"using device: {device}")

# torch.manual_seed(1162003)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1162003)

# --------------------------------
# batch size now is 32*512 = 16384
# gradient accumulation --> want a batch size of ~0.5M
total_batch_size = 131072 # 2**16
B = 64
assert total_batch_size % B == 0, "Batch size must be divisible by B*T*(world_size)"
grad_accum = total_batch_size // B

# --------------------------------
# ploting data
loss_plot = []
val_loss_plot = []
val_y_axis = []
norm_plot = []

def get_val_loss(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def training_pipeline():
    # if master_process:
    print(f"Total batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum}")

    config = ModelConfig()
    model = DaiJazz_Piano(config)
    model.to(device)
    num_param = sum(p.numel() for p in model.parameters())/1e6
    print(f"Generated model with: {num_param}M params")
    print(f'{len(train_loader.dataset)}: training samples')
    print(f'{len(val_loader.dataset)}: validation samples')
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    max_steps = len(train_loader) // grad_accum

    print(f"{max_steps} steps for 1 epoch")
    warmup_steps = max_steps // 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    start_time = time.time()
    it = 0
    previous_val_loss = None
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(1):
        loss_accum = 0
        # mini_batch = 0
        for i, (x, y) in enumerate(train_loader):
            # print(f"Step {it}, mini_batch {mini_batch}")
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum
            loss_accum += loss.detach()
            loss.backward()
            # mini_batch += 1

            if (i+1) % grad_accum == 0 or i == len(train_loader) - 1:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = get_lr(it, max_lr, min_lr, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                print(f'Iteration {it}, Loss {loss_accum.item()}, Average time {(end_time - start_time) * 1000 / 10}ms, Norm {norm:.4f}, LR {lr}')
                loss_accum = 0 # reset loss accumulator
                start_time = time.time()
                loss_plot.append(loss.item())
                norm_plot.append(norm)

                if it % 10 == 0 and it != 0 or it == max_steps - 1:
                    val_loss = get_val_loss(model, val_loader)
                    val_loss_plot.append(val_loss)
                    val_y_axis.append(it)
                    print(f'Validation loss: {val_loss}')
                    if previous_val_loss is None or previous_val_loss >= val_loss:
                        previous_val_loss = val_loss
                        print('Saving best model...')
                        torch.save(model.state_dict(), 'best_v6.pth')

                    print('Saving last model...')
                    torch.save(model.state_dict(), 'last_v6.pth')
                    model.train()
                # mini_batch = 0
                it += 1
    
    save_dir = 'plots'
    plot_save(loss_plot, 'Training Loss', 'Iterations', 'Loss', f'{save_dir}/loss_plot_v6.png')
    plot_save(val_loss_plot, 'Validation Loss', 'Iterations', 'Loss', f'{save_dir}/val_loss_plot_v6.png', val_y_axis)
    plot_save(norm_plot, 'Gradient Norm', 'Iterations', 'Norm', f'{save_dir}/norm_plot_v6.png')


if __name__ == '__main__':
    executor = submitit.AutoExecutor(
        folder='/Utilisateurs/tnguye38/v6_test')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        job_name='v6',
        timeout_min=2160 * 4,
        gpus_per_node=1,
        cpus_per_task=5,
        mem_gb=40,
        slurm_partition='gpu-a40',
        slurm_additional_parameters={
            'nodelist': 'l3icalcul05'
        }
    )
    executor.submit(training_pipeline)