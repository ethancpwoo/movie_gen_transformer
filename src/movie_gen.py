"""
Movie Summary Generator from transformer architecture. Using Wikipedia movie summaries from Kaggle.

Following guidelines from ShakespeareGPT by Andrej Karpathy and Attention Is All You Need paper. 
"""

import torch
import torch.nn as nn
import re
from torch.nn import functional as F


batch_size = 64 # how many sequences will be processed in parallel
block_size = 256 # how many tokens/nodes are we taking into context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 200
eval_iters = 200
head_size = 16 #size of attention head
learning_rate = 3e-4 # small network so really aggresive learning rate
max_iters = 5000
n_embd = 384
n_heads = 6
n_layers = 6

with open('summaries.txt', 'r', encoding='utf8') as f:
        text = f.read()

# ensure that there are only Latin and special character wording. CJK characters are removed
pattern = re.compile(r'[^\x00-\x7F0-9\[\]]+')
text = pattern.sub('', text)

# find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

enc_map = {}
dec_map = {}

for i, character in enumerate(chars):
    enc_map[character] = i
    dec_map[i] = character

def encode(s : str, enc_map : dict) -> list:
    ls = []
    for char in s:
        ls.append(enc_map[char])
    return ls

def decode(ls : list, dec_map : dict) -> str:
    char_list = []
    ls = ls.tolist()
    for i in ls: 
        char_list.append(dec_map[i]) 
    s = ''.join(char_list)
    return s

def set_data(text : list, enc_map: dict):
    data = torch.tensor(encode(text, enc_map), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data

# generates a small batch of data of inputs x and targets y
    
def get_batch(split : str, train_data : list, test_data : list):
    if split == 'train':
        data = train_data
    else: 
        data = test_data

    # Gets random position to grab a block of data, batch size number of random offsets
    # ix is 4 randomly generated numbers between 0 and len(data) - blocksize
        
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # stack all 1D tensors into batch size by block size tensor
    x = torch.stack([data[i:i+block_size] for i in ix])

    # y is 1 index ahead of x since y trains of all previous context x
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

class MovieModel(nn.Module):

    def __init__(self, vocab_size):
        super.__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print(device)
train_data, test_data = set_data(text, enc_map)

model = MovieModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses= estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train', train_data, test_data)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

rand_context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(rand_context, max_new_tokens=200)[0].tolist(), dec_map))