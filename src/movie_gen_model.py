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
    # y is 1 ahead of x since y trains of all previous context x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

#class Head(nn.Module):

#class MultiHeadAttention(nn.Module):

#class FeedForwardNetwork(nn.Module):

#class Block(nn.Module):

#class MovieGenerativeTransformer(nn.Module):

def main():
    text, enc_map, dec_map = read_text()



if __name__ == '__main__':
    main()