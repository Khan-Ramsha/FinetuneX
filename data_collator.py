

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

def pad(sequences, padding_value: int = 0, padding_side: str = "right", pad_to_multiple_of: int | None = None):

    assert padding_side in ("right", "left")

    max_len = max(seq.size(0) for seq in sequences)

    # round up to multiple
    if pad_to_multiple_of is not None:
        remainder = max_len % pad_to_multiple_of
        if remainder != 0:
            max_len += pad_to_multiple_of - remainder

    batch_size = len(sequences)
    device = sequences[0].device
    dtype = sequences[0].dtype

    padded = torch.full(
        (batch_size, max_len),
        padding_value,
        dtype=dtype,
        device=device,
    )

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            padded[i, :length] = seq
        else:
            padded[i, max_len - length:] = seq

    return padded

class DataCollator:
    def __init__(self, pad_token_id: int, completion_only_loss: bool = False):
        self.pad_token_id = pad_token_id
        self.completion_only_loss = completion_only_loss

    def __call__(self, examples):
        # Convert input_ids to tensors
        input_ids = [torch.tensor(e['input_ids'], dtype=torch.long) for e in examples]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        labels = [ids.clone() for ids in input_ids]

        if self.completion_only_loss:
            completion_mask = [torch.tensor(e["completion_mask"], dtype=torch.long) for e in examples]
        # Pad it
        input_ids = pad(input_ids, padding_value=self.pad_token_id, padding_side = "right", pad_to_multiple_of = 128)
        attention_mask = pad(attention_mask, padding_value = 0, padding_side = "right", pad_to_multiple_of = 128)
        labels = pad(labels, padding_value=-100, padding_side = "right", pad_to_multiple_of = 128)

        if self.completion_only_loss:
            completion_mask = pad(completion_mask, padding_value = 0, padding_side ="right", pad_to_multiple_of = 128)
            labels = labels.masked_fill_(completion_mask == 0, -100)

        for i, lbl in enumerate(labels):
            valid_tokens = (lbl != -100).sum().item() #tokens to compute loss on
            if valid_tokens == 0:
                print(f"ERROR: Sample {i} has all labels = -100!")
                raise ValueError(f"Sample {i} has no valid labels for training!")
            # else:
                # print(f"Sample {i}: {valid_tokens} valid tokens out of {len(lbl)}")

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return batch