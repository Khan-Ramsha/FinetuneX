import torch
from finetunex.text_generation.decoding import stochastic_sampling
@torch.inference_mode()
def generate(model,prompt, max_new_tokens, top_k, top_p, temperature, stop_tokens):
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    T = prompt.size(1) #seq len
    buffer_len = max((len(tokens) for tokens in stop_tokens), default = 1)
    yield_i = 0
    tokens = []
    curr = prompt
    for i in range(1, max_new_tokens - T + 1):
        output = model(curr)
        logits = output['logits']
        new_token = stochastic_sampling(logits, top_k=top_k, top_p=top_p, temp=temperature)
        tokens.append(new_token)
        new_token = new_token.view(1, 1)
        print(f"curr shape: {curr.shape}")
        print(f"new_token after view(1,1): {new_token.shape}")  
        print(f"new_token after unsqueeze(0): {new_token.unsqueeze(0).shape}")
        curr = torch.cat((curr, new_token), dim = -1)
        for st in stop_tokens:
            l = len(st)
            if l <= len(tokens):
                if all(a == b for a, b in zip(tokens[-l:], st)):
                    return
        if i - yield_i >= buffer_len:
            yield from tokens[yield_i:i]
            yield_i = i