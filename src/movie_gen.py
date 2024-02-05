"""
Movie Summary Generator from transformer architecture. Using Wikipedia movie summaries from Kaggle.

Following guidelines from ShakespeareGPT by Andrej Karpathy, Attention Is All You Need paper, and Dropout: A Simple Way to Prevent Overfitting. 

Classes: 
    Block: Standard block of transformer which includes attention and feedforward network.
    Head: Head of attention. Provides scaled dot product of positions of token context to predict.
    MultiHeadAttention: Multiple heads in parallel for increased prediction. 
    FeedForwardNetwork: Pass through feed forward to "learn" from context.
    MovieModel: Decoder architecture of transformer.
"""


import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import re

from matplotlib import style
from torch.nn import functional as F
from tqdm import tqdm


batch_size = 64 # how many sequences will be processed in parallel
block_size = 256 # how many tokens/nodes are we taking into context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2 #to prevent overfitting, drops out random neurons 
eval_interval = 200
eval_iters = 200
head_size = 16 #size of attention head
learning_rate = 3e-4 # small network so really aggresive learning rate
max_iters = 8000
n_embd = 384 # number of embedding dimensions, instead of going vocab -> logits, vocab_size -> n_embd -> logits for more params.
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
    """Encodes a string to token integer."""
    ls = []
    for char in s:
        ls.append(enc_map[char])
    return ls

def decode(ls : list, dec_map : dict) -> str:
    """Decodes a token integer to a string."""
    char_list = []
    for i in ls: 
        char_list.append(dec_map[i]) 
    s = ''.join(char_list)
    return s

def set_data(text : list, enc_map: dict):
    """Sets given text to tensor and divides data to train and eval."""
    data = torch.tensor(encode(text, enc_map), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data

def create_acc_loss_graph():
    """Creates loss graph for visual of loss."""
    style.use("ggplot")
    contents = open("out/modeldropout8000iter.log", "r").read().split("\n")
    iters = []
    val_accs = []
    val_losses = []

    for c in contents[:-1]: 
        name, iter, val_acc, val_loss = c.split(",")
        iters.append(float(iter))
        val_accs.append(float(val_acc))
        val_losses.append(float(val_loss))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0))

    ax1.plot(iters, val_accs, label="val_acc")
    ax1.plot(iters, val_losses, label="val_loss")
    ax1.legend(loc="upper left")
    ax1.set_xlabel("iteration")

    plt.savefig('out/losschartdropout8000iter')

# generates a small batch of data of inputs x and targets y
    
def get_batch(split : str, train_data : list, test_data : list):
    """Creates random batch of words from data to train/eval."""
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
def estimate_loss(train_data : list, test_data : list):
    """Used to estimate loss during training loops. Will not train on network."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, test_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Single head of self attention."""

    def __init__(self, head_size):
        """Linear functions as K, Q, V values. Creates triangle matrix and dropout."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # size = (B, T, C).
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # triangle matrix (block size, block size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Passes x through attention block. Returns tensor of affinity for each token."""

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
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #makes this a decoder attention block

        #softmax is normalization function which defines summing and meaning.
        wei = F.softmax(wei, dim=-1)

        # v are the elements we aggregate, not raw x. X is sort of private to this token.
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Stacked attention modules."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForwardNetwork(nn.Module):
    """Feed forward network after put through multiheadattention."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A block of transformer module."""
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
        return x

class EncoderLayer(nn.Module):
    """Not yet implemented."""
    def __init__(self, dim, num_heads, n_embd):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_foward = FeedForwardNetwork(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        atn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(atn_out))
        x = self.feed_foward(x)
        x = self.norm2(x + self.dropout(x))
        return x


class MovieModel(nn.Module):
    """Transformer decoder model. Combination of previous classes."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_embd)
        self.language_modeling_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        """
        Takes idx and trains with self-attention.

        Definitions:
        logits: outputs of nn before activation function
        tokens: inputs for nn
        """
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.layernorm(x)
        
        logits = self.language_modeling_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Takes idx which is randomly generated and generates new tokens based off context."""
        for _ in range(max_new_tokens):
            # crop idx to block size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on last step only
            logits = logits[:, -1, :]
            # softmax
            probs = F.softmax(logits, dim=-1)
            # sample from distrobution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the focused idx to the previous sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

train_data, test_data = set_data(text, enc_map)

model = MovieModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

with open("out/modeldropout8000iter.log", "a") as f:
    for iter in tqdm(range(max_iters)):

        if iter % eval_interval == 0:
            losses = estimate_loss(train_data, test_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            f.write(f"moviegendropout8000iter, {iter}, {losses['train']}, {losses['val']}\n")
        
        xb, yb = get_batch('train', train_data, test_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

rand_context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(rand_context, max_new_tokens=1000)[0].tolist(), dec_map))

create_acc_loss_graph()
torch.save(model.state_dict(), 'out/moviegendropout8000iter.pt')