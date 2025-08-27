import torch
from finetunex.text_generation.decoding import stochastic_sampling

def generate(model, inputs, max_new_tokens, top_p, top_k, temperature, stop_tokens):
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    batch, input_len = inputs.shape
    max_tokens = model.config.max_position_embeddings
    max_new_tokens = min(max_new_tokens, max_tokens - input_len - 10)
    curr_sequence = inputs
    generated = []
    for _ in range(max_new_tokens):
        if curr_sequence.size(1) >= max_tokens - 1:
            keep_len = max_tokens // 2
            curr_sequence = curr_sequence[:, -keep_len:]

        output = model(curr_sequence)
        logits = output['logits'][:, -1, :]
        next_token = stochastic_sampling(logits, temperature, top_k, top_p)
        if next_token.item() in stop_tokens:
            break
        generated.append(next_token.item())
        curr_sequence = torch.cat([curr_sequence, next_token], dim=1)
    return generated