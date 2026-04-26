import torch
from torch.utils.data import DataLoader
import math

def evaluate_model(model, dataset, data_collator, batch_size, device):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attn_mask=attention_mask, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * n_tokens
            total_tokens += n_tokens
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss, 
        "perplexity": perplexity
    }