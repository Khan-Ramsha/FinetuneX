import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataset, data_collator, batch_size, accelerator):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    data_loader = accelerator.prepare(data_loader)
    num_batches = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, labels=labels)
            loss = accelerator.gather(outputs['loss']).mean()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accelerator.print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    return avg_loss