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
head_size = 16 #size of attention head
learning_rate = 3e-4 # small network so really aggresive learning rate
max_iters = 5000
n_embd = 384
n_heads = 6
n_layers = 6
vocab_size = 0

def read_text():
    global vocab_size
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
    
    return text, enc_map, dec_map

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

    return x, y

class Head(nn.Module):

    # Single head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # size = (B, T, C).
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
    
    def forward(self, x):

        # Instead of summing the values in the tensor, now will have a query and a key.
        # Query is what I am looking for, Key is what I contain in terms of weight.
        # Ex: if I am vowel, my key will be align well with query of constanants will have a high affinity.

        B, T, C = x.shape

        # Single Head of self-attention (normally chat-gpt will have mutliple heads for increased accuracy for attention)
        # linear transformation template of y = x(A (transpose) ) + b
        # independently generated keys and query so they do not have any affinity yet. 

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        #here is the dot product to see which of the values generate affinity
        # Affinity between tokens in tensor, dot of key and query = wei.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5# due to batch dimension. (B, T, 16) @ (B, 16, T) => (B, T, T)

        tril = torch.tril(torch.ones(T, T)) 
        # triangle matrix (T, T)

        wei = wei.masked_fill(tril == 0, float('-inf'))

        #softmax is normalization function which defines summing and meaning.
        wei = F.softmax(wei, dim=-1)

        # v are the elements we aggregate, not raw x. X is sort of private to this token.
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    # Stacked attention modules
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out

class FeedForwardNetwork(nn.Module):
    # Feed forward network after put through multiheadattention
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # A block of transformer module
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.forwardnet = FeedForwardNetwork(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.self_attention(self.norm1(x)) # attention + residual connection
        x = x + self.forwardnet(self.norm2(x))


class MovieGenerativeTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        # For fast lookup, use embedding table for tokens
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.norm_layer = nn.LayerNorm(n_embd)
        self.linear_head = nn.Linear(n_embd, vocab_size)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both shapes (B, T) tensor of integers
        token_emb_table = self.token_embedding_table(idx)
        pos_emb_table = self.position_embedding_table(torch.arange(T, device=device))

        x = token_emb_table + pos_emb_table
        x = self.blocks(x)
        x = self.norm_layer(x)
        logits = self.linear_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def main():
    text, enc_map, dec_map = read_text()
    
    model = MovieGenerativeTransformer()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_data, test_data =  set_data(text, enc_map)

    print(vocab_size)

    for iter in range(max_iters):

        #if iter % eval_interval == 0:
            # get losses
        
        xb, yb = get_batch('train', train_data, test_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(iter)
    
    rand_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(rand_context, max_new_tokens=200)[0].tolist(), dec_map))


if __name__ == '__main__':
    main()