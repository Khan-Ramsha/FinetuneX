import torch
from finetunex.text_generation.decoding import stochastic_sampling

def generate(model, idx, max_new_tokens, context_size, temperature ,top_k, top_p, stop_tokens):
    model.eval()
    ctx_len = context_size

    with torch.no_grad():
        for _ in range(max_new_tokens):
            model_input = idx[:, -ctx_len:] if idx.size(1) > ctx_len else idx
            output = model(model_input)
            logits = output['logits']
            next_idx = stochastic_sampling(logits, temperature, top_k, top_p)
            if next_idx.item() in stop_tokens:
                break
            idx = torch.cat([idx, next_idx], dim=1)

    return idx