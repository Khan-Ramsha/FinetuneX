import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

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
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        if self.completion_only_loss:
            completion_mask = pad_sequence(completion_mask, batch_first=True, padding_value=0)
            labels = labels.masked_fill_(completion_mask == 0, -100)

        for i, lbl in enumerate(labels):
            valid_tokens = (lbl != -100).sum().item() #tokens to compute loss on
            if valid_tokens == 0:
                print(f"ERROR: Sample {i} has all labels = -100!")
                raise ValueError(f"Sample {i} has no valid labels for training!")
            else:
                print(f"Sample {i}: {valid_tokens} valid tokens out of {len(lbl)}")

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return batch
           
# collator = DataCollator(pad_token_id = 0, completion_only_loss = True)
# example = [
#     {
#         "input_ids": [101, 2023, 2003, 2307],  
#         "completion_mask": [0, 0, 1, 1],  
#     },
#     {
#         "input_ids": [101, 2009, 2003, 1037, 2204, 2154],
#         "completion_mask": [0, 0, 0, 1, 1, 1]
#     }
# ]
# result = collator(example)
# print(result)