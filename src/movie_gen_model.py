import torch
import torch.nn as nn
import re

from torch.nn import functional as F


# Movie Summary Generator from transformer architecture. Using Wikipedia movie summaries from Kaggle.

# Following guidelines from ShakespeareGPT by Andrej Karpathy

# First read in the entire dataset

def read_text():
    with open('summaries.txt', 'r', encoding='utf8') as f:
        text = f.read()

    #ensure that there are only Latin and special character wording. CJK characters are removed.
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

#def get_batch():

#class Head(nn.Module):

#class MultiHeadAttention(nn.Module):

#class FeedForwardNetwork(nn.Module):

#class Block(nn.Module):

#class MovieGenerativeTransformer(nn.Module):

def main():
    text, enc_map, dec_map = read_text()



if __name__ == '__main__':
    main()