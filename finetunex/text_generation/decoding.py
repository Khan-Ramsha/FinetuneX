""""This file implements Decoding strategies in text generation"""
import torch
import torch.nn.functional as F

def stochastic_sampling(logits, temp, top_k, top_p):
    logits = logits[:, -1, :]
    logits = logits / temp
    if top_k is not None:
        masked_logits = top_k_sampling(logits, top_k)
    
    if top_p is not None:
        masked_logits = top_p_sampling(logits, top_p)

    probabilities = torch.softmax(masked_logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, num_samples=1)
    return sampled_token

def top_k_sampling(logits, k):
    top_k, idx = torch.topk(logits, k, dim = -1)
    mask = torch.full_like(logits, float("-inf"))
    masked_logits = mask.scatter_(-1, idx, top_k)
    return masked_logits

def top_p_sampling(logits, top_p_val):
    sorted_logits, idx = torch.sort(logits, descending=True, dim = -1)#sorting by probab
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) #cumulative sum
    remove = cumulative_probs > top_p_val #exclude rest- tokens beyond the nucleus set 
    remove[...,1: ] = remove[..., : -1].clone() # shifting right
    remove[...,0] = False #if highest probab is top_p itself, no tokens will be selected. so keep atleast first. 
    indx = torch.zeros_like(logits, dtype = torch.bool)
    indx.scatter_(-1, idx, remove) # indices as per orginal vocab
    logits = logits.masked_fill(indx, float("-inf"))
    return logits