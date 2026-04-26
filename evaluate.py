import torch
from torch.utils.data import DataLoader
import math
from utils import token_accuracy

def evaluate_model(model, dataset, data_collator, batch_size, device):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    total_loss = 0.0
    total_correct = 0
    total_supervised = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attn_mask=attention_mask, labels=labels)
            correct, total_sup = token_accuracy(outputs["logits"], labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * n_tokens
            total_tokens += n_tokens
            total_correct += correct
            total_supervised += total_sup
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)
        token_acc =  total_correct/max(total_supervised, 1)

    return {
        "loss": avg_loss, 
        "perplexity": perplexity,
        "token_accuracy": token_acc
    }