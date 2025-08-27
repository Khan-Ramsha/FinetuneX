""""This file implements Decoding strategies in text generation"""
import torch
import torch.nn.functional as F

def stochastic_sampling(logits, temp, top_k, top_p):
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # Take last position
    
    if temp > 0:
        logits = logits / temp #temperature scaling
    if top_k is not None:
        logits = top_k_sampling(logits, top_k)
    
    if top_p is not None:
        logits = top_p_sampling(logits, top_p)

    probabilities = torch.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, num_samples=1)
    return sampled_token

def top_k_sampling(logits, k):
    top_k, idx = torch.topk(logits, k, dim = -1) #since logits [B, T, vocab] so applyinig topk on last dim of logits that is vocab
    mask = torch.full_like(logits, float("-inf"))
    masked_logits = mask.scatter_(-1, idx, top_k) #-1 for last dim (vocab here)
    return masked_logits

def top_p_sampling(logits, top_p_val):
    sorted_logits, idx = torch.sort(logits, descending=True, dim = -1)#sorting by probab
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) #cumulative sum
    keep_mask = cumulative_probs <= top_p_val #exclude rest- tokens beyond the nucleus set 
    keep_mask[..., 0] = True
    mask = torch.zeros_like(logits, dtype = torch.bool)
    mask.scatter_(-1, idx, keep_mask)
    #now filter out unwanted tokens (outside of nucleus set)
    remove_mask = ~mask #contains true for tokens to be ignored
    logits = logits.masked_fill(remove_mask, float("-inf"))
    return logits