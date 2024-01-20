"""
Movie Summary Generator from transformer architecture. Using Wikipedia movie summaries from Kaggle.

Following guidelines from ShakespeareGPT by Andrej Karpathy
"""

import torch
import torch.nn as nn
import re
from torch.nn import functional as F


test_data = []
train_data = []
batch_size = 8 # how many sequences will be processed in parallel
block_size = 256 # how many tokens/nodes are we taking into context
head_size = 16 #size of attention head

def read_text():
    with open('summaries.txt', 'r', encoding='utf8') as f:
        text = f.read()

    # ensure that there are only Latin and special character wording. CJK characters are removed
    pattern = re.compile(r'[^\x00-\x7F0-9\[\]]+')
    text = pattern.sub('', text)

    # find all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)

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
    for i in ls: 
        char_list.append(dec_map[i]) 
    s = ''.join(char_list)
    return s

def set_data(text : list):
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.type)

    #print first 1000 tokens from tensor
    print(data[:1000])

    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

# generates a small batch of data of inputs x and targets y
    
def get_batch(split : str):
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
        self.key = nn.Linear(C, head_size, bias=False) # size = (B, T, 16).
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
    
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

class FeedForwardNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()

#class Block(nn.Module):
    
class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

#class MovieGenerativeTransformer(nn.Module):

def main():
    text, enc_map, dec_map = read_text()



if __name__ == '__main__':
    main()