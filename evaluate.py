import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate_model(model, dataset, data_collator, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    num_batches = 0
    total_loss = 0.0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
